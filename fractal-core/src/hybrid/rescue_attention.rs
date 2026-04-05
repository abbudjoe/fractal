use burn::{
    config::Config,
    module::Module,
    nn::Initializer,
    prelude::ElementConversion,
    tensor::{activation::softmax, backend::Backend, Bool, Int, Tensor},
};

use crate::{
    error::FractalError,
    projection::{ProjectionLayoutPolicy, StructuredProjection, StructuredProjectionConfig},
};

use super::{
    model::HybridRescuePrevalidationMode,
    retrieval_gather::{
        GatheredRetrievalContext, GatheredRetrievalLayout, GatheredRetrievalProvenance,
    },
};

const RESCUE_ATTENTION_INIT_MIN: f64 = -0.08;
const RESCUE_ATTENTION_INIT_MAX: f64 = 0.08;
const MASKED_SCORE_FLOOR: f64 = -1.0e9;
pub const PHASE1_LOCAL_WINDOW_SIZE: usize = 256;
pub const PHASE1_ROUTED_SPAN_COUNT: usize = 8;
pub const PHASE1_LEAF_SIZE: usize = 16;
pub const PHASE1_REMOTE_TOKEN_BUDGET: usize = PHASE1_ROUTED_SPAN_COUNT * PHASE1_LEAF_SIZE;
pub const PHASE1_TOTAL_TOKEN_BUDGET: usize =
    PHASE1_LOCAL_WINDOW_SIZE + PHASE1_REMOTE_TOKEN_BUDGET;

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct RescueAttentionShape {
    pub token_state_dim: usize,
    pub attention_dim: usize,
    pub local_window_size: usize,
    pub routed_span_count: usize,
    pub leaf_size: usize,
    pub remote_token_budget: usize,
    pub sink_token_count: usize,
    pub total_token_budget: usize,
}

impl RescueAttentionShape {
    pub fn validate(self) -> Result<Self, FractalError> {
        ensure_nonzero("rescue_attention.token_state_dim", self.token_state_dim)?;
        ensure_nonzero("rescue_attention.attention_dim", self.attention_dim)?;
        ensure_nonzero("rescue_attention.local_window_size", self.local_window_size)?;
        ensure_nonzero("rescue_attention.routed_span_count", self.routed_span_count)?;
        ensure_nonzero("rescue_attention.leaf_size", self.leaf_size)?;
        ensure_nonzero(
            "rescue_attention.remote_token_budget",
            self.remote_token_budget,
        )?;
        ensure_nonzero(
            "rescue_attention.total_token_budget",
            self.total_token_budget,
        )?;

        let expected_remote_budget = self
            .routed_span_count
            .checked_mul(self.leaf_size)
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "rescue_attention routed span budget overflowed".to_string(),
                )
            })?;
        ensure_match(
            "rescue_attention.remote_token_budget",
            self.remote_token_budget,
            expected_remote_budget,
        )?;

        let combined_budget = self
            .local_window_size
            .checked_add(self.remote_token_budget)
            .and_then(|value| value.checked_add(self.sink_token_count))
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "rescue_attention total token budget overflowed".to_string(),
                )
            })?;
        ensure_match(
            "rescue_attention.total_token_budget",
            combined_budget,
            self.total_token_budget,
        )?;

        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RescueAttentionDiagnostics {
    pub mean_local_attention_mass: f32,
    pub mean_remote_attention_mass: f32,
}

#[derive(Debug, Clone)]
pub struct RescueAttentionInput<B: Backend> {
    mode: HybridRescuePrevalidationMode,
    query_state: Tensor<B, 2>,
    query_positions: Tensor<B, 1, Int>,
    local_token_states: Tensor<B, 3>,
    local_token_positions: Tensor<B, 2, Int>,
    local_token_mask: Tensor<B, 2, Bool>,
    gathered_remote: GatheredRetrievalContext<B>,
}

impl<B: Backend> RescueAttentionInput<B> {
    pub fn new(
        mode: HybridRescuePrevalidationMode,
        query_state: Tensor<B, 2>,
        query_positions: Tensor<B, 1, Int>,
        local_token_states: Tensor<B, 3>,
        local_token_positions: Tensor<B, 2, Int>,
        local_token_mask: Tensor<B, 2, Bool>,
        gathered_remote: GatheredRetrievalContext<B>,
    ) -> Result<Self, FractalError> {
        let [batch_size, token_state_dim] = query_state.dims();
        ensure_nonzero("rescue_attention_input.batch_size", batch_size)?;
        ensure_nonzero("rescue_attention_input.token_state_dim", token_state_dim)?;
        ensure_dims1(
            "rescue_attention_input.query_positions",
            query_positions.dims(),
            [batch_size],
        )?;

        let [local_batch_size, local_token_count, local_token_state_dim] =
            local_token_states.dims();
        ensure_match(
            "rescue_attention_input.local_token_states.batch_size",
            local_batch_size,
            batch_size,
        )?;
        ensure_nonzero(
            "rescue_attention_input.local_token_states.token_count",
            local_token_count,
        )?;
        ensure_match(
            "rescue_attention_input.local_token_states.token_state_dim",
            local_token_state_dim,
            token_state_dim,
        )?;
        ensure_dims2(
            "rescue_attention_input.local_token_positions",
            local_token_positions.dims(),
            [batch_size, local_token_count],
        )?;
        ensure_dims2(
            "rescue_attention_input.local_token_mask",
            local_token_mask.dims(),
            [batch_size, local_token_count],
        )?;
        ensure_match(
            "rescue_attention_input.gathered_remote.batch_size",
            gathered_remote.shape().batch_size,
            batch_size,
        )?;
        ensure_match(
            "rescue_attention_input.gathered_remote.token_state_dim",
            gathered_remote.shape().token_state_dim,
            token_state_dim,
        )?;

        Ok(Self {
            mode,
            query_state,
            query_positions,
            local_token_states,
            local_token_positions,
            local_token_mask,
            gathered_remote,
        })
    }

    pub fn mode(&self) -> HybridRescuePrevalidationMode {
        self.mode
    }

    pub fn query_state(&self) -> Tensor<B, 2> {
        self.query_state.clone()
    }

    pub fn query_positions(&self) -> Tensor<B, 1, Int> {
        self.query_positions.clone()
    }

    pub fn local_token_states(&self) -> Tensor<B, 3> {
        self.local_token_states.clone()
    }

    pub fn local_token_positions(&self) -> Tensor<B, 2, Int> {
        self.local_token_positions.clone()
    }

    pub fn local_token_mask(&self) -> Tensor<B, 2, Bool> {
        self.local_token_mask.clone()
    }

    pub fn gathered_remote(&self) -> &GatheredRetrievalContext<B> {
        &self.gathered_remote
    }
}

#[derive(Debug, Clone)]
pub struct RescueAttentionOutput<B: Backend> {
    updated_state: Tensor<B, 2>,
    attention_weights: Tensor<B, 2>,
    diagnostics: RescueAttentionDiagnostics,
}

impl<B: Backend> RescueAttentionOutput<B> {
    pub fn new(
        updated_state: Tensor<B, 2>,
        attention_weights: Tensor<B, 2>,
        diagnostics: RescueAttentionDiagnostics,
    ) -> Result<Self, FractalError> {
        let [batch_size, token_state_dim] = updated_state.dims();
        ensure_nonzero("rescue_attention_output.batch_size", batch_size)?;
        ensure_nonzero("rescue_attention_output.token_state_dim", token_state_dim)?;
        let [weight_batch_size, total_context_tokens] = attention_weights.dims();
        ensure_match(
            "rescue_attention_output.attention_weights.batch_size",
            weight_batch_size,
            batch_size,
        )?;
        ensure_nonzero(
            "rescue_attention_output.attention_weights.total_context_tokens",
            total_context_tokens,
        )?;

        Ok(Self {
            updated_state,
            attention_weights,
            diagnostics,
        })
    }

    pub fn updated_state(&self) -> Tensor<B, 2> {
        self.updated_state.clone()
    }

    pub fn attention_weights(&self) -> Tensor<B, 2> {
        self.attention_weights.clone()
    }

    pub fn diagnostics(&self) -> &RescueAttentionDiagnostics {
        &self.diagnostics
    }
}

pub trait RescueAttentionBlock<B: Backend>: Module<B> + core::fmt::Debug {
    fn shape(&self) -> RescueAttentionShape;

    fn attend(
        &self,
        input: RescueAttentionInput<B>,
    ) -> Result<RescueAttentionOutput<B>, FractalError>;
}

#[derive(Config, Debug)]
pub struct BaselineRescueAttentionConfig {
    pub token_state_dim: usize,
    pub attention_dim: usize,
    pub local_window_size: usize,
    pub routed_span_count: usize,
    pub leaf_size: usize,
    pub sink_token_count: usize,
    pub total_token_budget: usize,
    #[config(
        default = "Initializer::Uniform { min: RESCUE_ATTENTION_INIT_MIN, max: RESCUE_ATTENTION_INIT_MAX }"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct BaselineRescueAttentionBlock<B: Backend> {
    query_projection: StructuredProjection<B>,
    key_projection: StructuredProjection<B>,
    value_projection: StructuredProjection<B>,
    shape: RescueAttentionShape,
}

impl BaselineRescueAttentionConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        let shape = RescueAttentionShape {
            token_state_dim: self.token_state_dim,
            attention_dim: self.attention_dim,
            local_window_size: self.local_window_size,
            routed_span_count: self.routed_span_count,
            leaf_size: self.leaf_size,
            remote_token_budget: self
                .routed_span_count
                .checked_mul(self.leaf_size)
                .ok_or_else(|| {
                    FractalError::InvalidConfig(
                        "baseline_rescue_attention.remote_token_budget overflowed".to_string(),
                    )
                })?,
            sink_token_count: self.sink_token_count,
            total_token_budget: self.total_token_budget,
        };
        shape.validate()?;
        if self.sink_token_count != 0 {
            return Err(FractalError::InvalidConfig(
                "baseline_rescue_attention.sink_token_count must be 0 in phase 1".to_string(),
            ));
        }
        ensure_match(
            "baseline_rescue_attention.local_window_size",
            self.local_window_size,
            PHASE1_LOCAL_WINDOW_SIZE,
        )?;
        ensure_match(
            "baseline_rescue_attention.routed_span_count",
            self.routed_span_count,
            PHASE1_ROUTED_SPAN_COUNT,
        )?;
        ensure_match(
            "baseline_rescue_attention.leaf_size",
            self.leaf_size,
            PHASE1_LEAF_SIZE,
        )?;
        ensure_match(
            "baseline_rescue_attention.total_token_budget",
            self.total_token_budget,
            PHASE1_TOTAL_TOKEN_BUDGET,
        )?;
        Ok(())
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineRescueAttentionBlock<B> {
        self.try_init(device).unwrap_or_else(|error| {
            panic!("invalid baseline rescue attention config: {error}");
        })
    }

    pub fn try_init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<BaselineRescueAttentionBlock<B>, FractalError> {
        self.validate()?;
        let projection = |d_input, d_output| -> StructuredProjection<B> {
            StructuredProjectionConfig::new(d_input, d_output)
                .with_layout_policy(ProjectionLayoutPolicy::OutputByInput)
                .with_initializer(self.initializer.clone())
                .init(device)
        };
        Ok(BaselineRescueAttentionBlock {
            query_projection: projection(self.token_state_dim, self.attention_dim),
            key_projection: projection(self.token_state_dim, self.attention_dim),
            value_projection: projection(self.token_state_dim, self.token_state_dim),
            shape: RescueAttentionShape {
                token_state_dim: self.token_state_dim,
                attention_dim: self.attention_dim,
                local_window_size: self.local_window_size,
                routed_span_count: self.routed_span_count,
                leaf_size: self.leaf_size,
                remote_token_budget: self.routed_span_count * self.leaf_size,
                sink_token_count: self.sink_token_count,
                total_token_budget: self.total_token_budget,
            },
        })
    }
}

impl<B: Backend> RescueAttentionBlock<B> for BaselineRescueAttentionBlock<B> {
    fn shape(&self) -> RescueAttentionShape {
        self.shape
    }

    fn attend(
        &self,
        input: RescueAttentionInput<B>,
    ) -> Result<RescueAttentionOutput<B>, FractalError> {
        validate_input_against_shape(&self.shape, &input)?;

        let [batch_size, local_token_count, _] = input.local_token_states().dims();
        let gathered_shape = input.gathered_remote().shape();
        let remote_token_count = gathered_shape.token_capacity;
        let total_context_tokens = local_token_count + remote_token_count;

        let query_projection = self.query_projection.forward(input.query_state());
        let local_keys =
            project_token_states(&self.key_projection, input.local_token_states(), batch_size)?;
        let local_values = project_token_states(
            &self.value_projection,
            input.local_token_states(),
            batch_size,
        )?;
        let remote_keys = project_token_states(
            &self.key_projection,
            input.gathered_remote().token_states(),
            batch_size,
        )?;
        let remote_values = project_token_states(
            &self.value_projection,
            input.gathered_remote().token_states(),
            batch_size,
        )?;

        let query_positions =
            tensor_data_to_i64(input.query_positions(), "rescue_attention.query_positions")?;
        let local_positions = tensor_data_to_i64(
            input.local_token_positions(),
            "rescue_attention.local_token_positions",
        )?;
        let local_mask = tensor_data_to_bool(
            input.local_token_mask(),
            "rescue_attention.local_token_mask",
        )?;
        let remote_positions = tensor_data_to_i64(
            input.gathered_remote().absolute_positions(),
            "rescue_attention.gathered_remote.absolute_positions",
        )?;
        let remote_mask = tensor_data_to_bool(
            input.gathered_remote().token_mask(),
            "rescue_attention.gathered_remote.token_mask",
        )?;
        let remote_span_ends = tensor_data_to_i64(
            input.gathered_remote().source_span_ends(),
            "rescue_attention.gathered_remote.source_span_ends",
        )?;

        let mut updated_rows = Vec::with_capacity(batch_size);
        let mut attention_rows = Vec::with_capacity(batch_size);
        let mut local_attention_mass_sum = 0.0f32;
        let mut remote_attention_mass_sum = 0.0f32;

        for (batch_index, query_position_raw) in query_positions.iter().enumerate().take(batch_size)
        {
            let query_position = usize::try_from(*query_position_raw).map_err(|_| {
                FractalError::InvalidState(
                    "rescue attention query position must be non-negative".to_string(),
                )
            })?;
            let active_local_token_count =
                count_batch_active_tokens(batch_index, local_token_count, &local_mask);
            ensure_causal_positions(
                input.mode(),
                batch_index,
                query_position,
                active_local_token_count,
                self.shape.local_window_size,
                self.shape.total_token_budget,
                local_token_count,
                remote_token_count,
                &local_positions,
                &local_mask,
                &remote_positions,
                &remote_span_ends,
                &remote_mask,
            )?;

            let query = query_projection
                .clone()
                .slice([batch_index..batch_index + 1, 0..self.shape.attention_dim])
                .reshape([1, self.shape.attention_dim]);
            let local_key_rows = local_keys
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    0..local_token_count,
                    0..self.shape.attention_dim,
                ])
                .reshape([local_token_count, self.shape.attention_dim]);
            let local_value_rows = local_values
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    0..local_token_count,
                    0..self.shape.token_state_dim,
                ])
                .reshape([local_token_count, self.shape.token_state_dim]);
            let remote_key_rows = remote_keys
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    0..remote_token_count,
                    0..self.shape.attention_dim,
                ])
                .reshape([remote_token_count, self.shape.attention_dim]);
            let remote_value_rows = remote_values
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    0..remote_token_count,
                    0..self.shape.token_state_dim,
                ])
                .reshape([remote_token_count, self.shape.token_state_dim]);

            let local_scores = score_rows(query.clone(), local_key_rows, self.shape.attention_dim);
            let remote_scores = score_rows(query, remote_key_rows, self.shape.attention_dim);
            let combined_scores = Tensor::cat(
                vec![
                    local_scores.reshape([1, local_token_count]),
                    remote_scores.reshape([1, remote_token_count]),
                ],
                1,
            )
            .reshape([total_context_tokens]);
            let combined_mask = combined_mask_tensor::<B>(
                batch_index,
                local_token_count,
                remote_token_count,
                &local_mask,
                &remote_mask,
                &combined_scores.device(),
            );
            ensure_true_visible_context(combined_mask.clone(), batch_index)?;
            let masked_scores =
                Tensor::<B, 1>::zeros([total_context_tokens], &combined_scores.device())
                    .add_scalar(MASKED_SCORE_FLOOR)
                    .mask_where(combined_mask, combined_scores);
            let attention = softmax(masked_scores.reshape([1, total_context_tokens]), 1)
                .reshape([total_context_tokens]);
            let combined_values = Tensor::cat(vec![local_value_rows, remote_value_rows], 0);
            let repeated_attention = attention
                .clone()
                .reshape([total_context_tokens, 1])
                .repeat(&[1, self.shape.token_state_dim]);
            let context = (combined_values * repeated_attention)
                .sum_dim(0)
                .reshape([1, self.shape.token_state_dim]);
            let original_query = input
                .query_state()
                .slice([batch_index..batch_index + 1, 0..self.shape.token_state_dim]);
            let updated_state = original_query + context;
            let attention_data = attention
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .map_err(|error| {
                    FractalError::InvalidState(format!(
                        "rescue_attention.attention_weights data conversion failed: {error}"
                    ))
                })?;
            let (local_mass, remote_mass) =
                summarize_attention_mass(&attention_data, local_token_count);
            local_attention_mass_sum += local_mass;
            remote_attention_mass_sum += remote_mass;
            updated_rows.push(updated_state);
            attention_rows.push(attention.reshape([1, total_context_tokens]));
        }

        RescueAttentionOutput::new(
            Tensor::cat(updated_rows, 0),
            Tensor::cat(attention_rows, 0),
            RescueAttentionDiagnostics {
                mean_local_attention_mass: local_attention_mass_sum / batch_size as f32,
                mean_remote_attention_mass: remote_attention_mass_sum / batch_size as f32,
            },
        )
    }
}

fn validate_input_against_shape<B: Backend>(
    shape: &RescueAttentionShape,
    input: &RescueAttentionInput<B>,
) -> Result<(), FractalError> {
    let [batch_size, token_state_dim] = input.query_state().dims();
    ensure_match(
        "rescue_attention.query_state.token_state_dim",
        token_state_dim,
        shape.token_state_dim,
    )?;
    let [local_batch_size, local_token_count, local_token_state_dim] =
        input.local_token_states().dims();
    ensure_match(
        "rescue_attention.local_token_states.batch_size",
        local_batch_size,
        batch_size,
    )?;
    ensure_match(
        "rescue_attention.local_token_states.token_state_dim",
        local_token_state_dim,
        shape.token_state_dim,
    )?;
    let gathered_shape = input.gathered_remote().shape();
    ensure_match(
        "rescue_attention.gathered_remote.batch_size",
        gathered_shape.batch_size,
        batch_size,
    )?;
    ensure_match(
        "rescue_attention.gathered_remote.token_state_dim",
        gathered_shape.token_state_dim,
        shape.token_state_dim,
    )?;
    let expected_layout = match input.mode() {
        HybridRescuePrevalidationMode::LocalOnly
        | HybridRescuePrevalidationMode::RoutedRemote
        | HybridRescuePrevalidationMode::OracleRemote => GatheredRetrievalLayout::SealedSpanPacks {
            max_span_count: shape.routed_span_count,
            leaf_size: shape.leaf_size,
        },
        HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset => {
            GatheredRetrievalLayout::ExactTokenSubset {
                max_token_count: shape.remote_token_budget,
            }
        }
    };
    let expected_provenance = match input.mode() {
        HybridRescuePrevalidationMode::LocalOnly => GatheredRetrievalProvenance::Suppressed,
        HybridRescuePrevalidationMode::RoutedRemote => GatheredRetrievalProvenance::Routed,
        HybridRescuePrevalidationMode::OracleRemote => GatheredRetrievalProvenance::Oracle,
        HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset => {
            GatheredRetrievalProvenance::OracleExactTokenSubset
        }
    };
    if gathered_shape.layout != expected_layout {
        return Err(FractalError::InvalidConfig(format!(
            "rescue_attention.gathered_remote.layout mismatch: expected {expected_layout:?}, got {:?}",
            gathered_shape.layout
        )));
    }
    if gathered_shape.provenance != expected_provenance {
        return Err(FractalError::InvalidConfig(format!(
            "rescue_attention.gathered_remote.provenance mismatch: expected {expected_provenance:?}, got {:?}",
            gathered_shape.provenance
        )));
    }

    let local_mask =
        tensor_data_to_bool(input.local_token_mask(), "rescue_attention.local_token_mask")?;
    let remote_active_counts = input.gathered_remote().active_token_counts()?;
    for (batch_index, remote_active_count) in remote_active_counts
        .iter()
        .copied()
        .enumerate()
        .take(batch_size)
    {
        let local_active_count =
            count_batch_active_tokens(batch_index, local_token_count, &local_mask);
        if remote_active_count > shape.remote_token_budget {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention batch {batch_index} gathered remote token count must be at most {}, got {}",
                shape.remote_token_budget, remote_active_count
            )));
        }
        if input.mode() == HybridRescuePrevalidationMode::LocalOnly && remote_active_count != 0 {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention batch {batch_index} local-only mode must not contain active remote tokens"
            )));
        }

        let local_budget_cap = if input.mode() == HybridRescuePrevalidationMode::LocalOnly {
            shape.total_token_budget
        } else {
            shape.local_window_size
        };
        if local_active_count > local_budget_cap {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention batch {batch_index} local token count must be at most {local_budget_cap}, got {local_active_count}"
            )));
        }

        let combined_budget = local_active_count
            .checked_add(remote_active_count)
            .and_then(|value| value.checked_add(shape.sink_token_count))
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "rescue_attention live attention budget overflowed".to_string(),
                )
            })?;
        if combined_budget > shape.total_token_budget {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention batch {batch_index} live attention budget must be at most {}, got {}",
                shape.total_token_budget, combined_budget
            )));
        }
    }

    Ok(())
}

fn project_token_states<B: Backend>(
    projection: &StructuredProjection<B>,
    token_states: Tensor<B, 3>,
    batch_size: usize,
) -> Result<Tensor<B, 3>, FractalError> {
    let [state_batch_size, token_count, token_state_dim] = token_states.dims();
    ensure_match(
        "rescue_attention.project_token_states.batch_size",
        state_batch_size,
        batch_size,
    )?;
    ensure_nonzero(
        "rescue_attention.project_token_states.token_count",
        token_count,
    )?;
    let projected_dim = projection.logical_dims()[1];
    Ok(projection
        .forward(token_states.reshape([batch_size * token_count, token_state_dim]))
        .reshape([batch_size, token_count, projected_dim]))
}

fn score_rows<B: Backend>(
    query: Tensor<B, 2>,
    keys: Tensor<B, 2>,
    attention_dim: usize,
) -> Tensor<B, 1> {
    let [token_count, _] = keys.dims();
    (keys * Tensor::cat(vec![query; token_count], 0))
        .sum_dim(1)
        .mul_scalar(1.0 / (attention_dim as f64).sqrt())
        .reshape([token_count])
}

fn count_batch_active_tokens(batch_index: usize, token_count: usize, mask: &[bool]) -> usize {
    let mut active_count = 0usize;
    for slot in 0..token_count {
        if mask[batch_index * token_count + slot] {
            active_count += 1;
        }
    }
    active_count
}

#[allow(clippy::too_many_arguments)]
fn ensure_causal_positions(
    mode: HybridRescuePrevalidationMode,
    batch_index: usize,
    query_position: usize,
    active_local_token_count: usize,
    local_window_size: usize,
    total_token_budget: usize,
    local_token_count: usize,
    remote_token_count: usize,
    local_positions: &[i64],
    local_mask: &[bool],
    remote_positions: &[i64],
    remote_span_ends: &[i64],
    remote_mask: &[bool],
) -> Result<(), FractalError> {
    let local_window_span = if mode == HybridRescuePrevalidationMode::LocalOnly {
        total_token_budget
    } else {
        local_window_size
    };
    let earliest_local_position = query_position.saturating_add(1).saturating_sub(local_window_span);

    for slot in 0..local_token_count {
        let flat_index = batch_index * local_token_count + slot;
        if !local_mask[flat_index] {
            continue;
        }
        let absolute_position = usize::try_from(local_positions[flat_index]).map_err(|_| {
            FractalError::InvalidConfig(
                "rescue_attention local token position must be non-negative".to_string(),
            )
        })?;
        if absolute_position > query_position {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention local token position {} must be less than or equal to query position {}",
                absolute_position, query_position
            )));
        }
        if absolute_position < earliest_local_position {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention local token position {} must lie within the last {} visible local tokens for query position {}",
                absolute_position, local_window_span, query_position
            )));
        }
    }

    for slot in 0..remote_token_count {
        let flat_index = batch_index * remote_token_count + slot;
        if !remote_mask[flat_index] {
            continue;
        }
        let absolute_position = usize::try_from(remote_positions[flat_index]).map_err(|_| {
            FractalError::InvalidConfig(
                "rescue_attention remote token position must be non-negative".to_string(),
            )
        })?;
        let span_end = usize::try_from(remote_span_ends[flat_index]).map_err(|_| {
            FractalError::InvalidConfig(
                "rescue_attention remote span end must be non-negative".to_string(),
            )
        })?;
        if span_end > query_position {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention remote span end {} must be less than or equal to query position {}",
                span_end, query_position
            )));
        }
        if absolute_position >= query_position {
            return Err(FractalError::InvalidConfig(format!(
                "rescue_attention remote token position {} must be strictly less than query position {}",
                absolute_position, query_position
            )));
        }
    }

    if active_local_token_count == 0 {
        return Err(FractalError::InvalidConfig(format!(
            "rescue_attention batch {batch_index} must contain at least one visible local token"
        )));
    }

    Ok(())
}

fn combined_mask_tensor<B: Backend>(
    batch_index: usize,
    local_token_count: usize,
    remote_token_count: usize,
    local_mask: &[bool],
    remote_mask: &[bool],
    device: &B::Device,
) -> Tensor<B, 1, Bool> {
    let mut values = Vec::with_capacity(local_token_count + remote_token_count);
    for slot in 0..local_token_count {
        values.push(local_mask[batch_index * local_token_count + slot]);
    }
    for slot in 0..remote_token_count {
        values.push(remote_mask[batch_index * remote_token_count + slot]);
    }

    Tensor::<B, 1, Bool>::from_data(values.as_slice(), device)
}

fn ensure_true_visible_context<B: Backend>(
    mask: Tensor<B, 1, Bool>,
    batch_index: usize,
) -> Result<(), FractalError> {
    if !mask.any().into_scalar().elem::<bool>() {
        return Err(FractalError::InvalidConfig(format!(
            "rescue_attention batch {batch_index} must contain at least one visible context token"
        )));
    }

    Ok(())
}

fn summarize_attention_mass(values: &[f32], local_token_count: usize) -> (f32, f32) {
    let local_mass = values.iter().take(local_token_count).sum();
    let remote_mass = values.iter().skip(local_token_count).sum();
    (local_mass, remote_mass)
}

fn tensor_data_to_i64<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Int>,
    name: &str,
) -> Result<Vec<i64>, FractalError> {
    tensor
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|error| {
            FractalError::InvalidState(format!("{name} data conversion failed: {error}"))
        })
}

fn tensor_data_to_bool<B: Backend, const D: usize>(
    tensor: Tensor<B, D, Bool>,
    name: &str,
) -> Result<Vec<bool>, FractalError> {
    tensor
        .to_data()
        .convert::<bool>()
        .into_vec::<bool>()
        .map_err(|error| {
            FractalError::InvalidState(format!("{name} data conversion failed: {error}"))
        })
}

fn ensure_nonzero(name: &str, value: usize) -> Result<(), FractalError> {
    if value == 0 {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be greater than zero"
        )));
    }

    Ok(())
}

fn ensure_match(name: &str, actual: usize, expected: usize) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected}, got {actual}"
        )));
    }

    Ok(())
}

fn ensure_dims1(name: &str, actual: [usize; 1], expected: [usize; 1]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected:?}, got {actual:?}"
        )));
    }

    Ok(())
}

fn ensure_dims2(name: &str, actual: [usize; 2], expected: [usize; 2]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected:?}, got {actual:?}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::tensor::backend::Backend;
    use burn::{
        backend::Candle,
        tensor::{Tensor, TensorData},
    };

    use super::*;

    type TestBackend = Candle<f32, i64>;

    fn baseline_block() -> BaselineRescueAttentionBlock<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        BaselineRescueAttentionConfig {
            token_state_dim: 4,
            attention_dim: 3,
            local_window_size: PHASE1_LOCAL_WINDOW_SIZE,
            routed_span_count: PHASE1_ROUTED_SPAN_COUNT,
            leaf_size: PHASE1_LEAF_SIZE,
            sink_token_count: 0,
            total_token_budget: PHASE1_TOTAL_TOKEN_BUDGET,
            initializer: Initializer::Constant { value: 0.25 },
        }
        .init::<TestBackend>(&device)
    }

    fn sealed_span_layout() -> GatheredRetrievalLayout {
        GatheredRetrievalLayout::SealedSpanPacks {
            max_span_count: PHASE1_ROUTED_SPAN_COUNT,
            leaf_size: PHASE1_LEAF_SIZE,
        }
    }

    fn exact_subset_layout() -> GatheredRetrievalLayout {
        GatheredRetrievalLayout::ExactTokenSubset {
            max_token_count: PHASE1_REMOTE_TOKEN_BUDGET,
        }
    }

    fn gathered_context(
        provenance: GatheredRetrievalProvenance,
        active_remote_tokens: usize,
    ) -> GatheredRetrievalContext<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        let token_capacity = PHASE1_REMOTE_TOKEN_BUDGET;
        let mut token_state_values = vec![0.0f32; token_capacity * 4];
        let mut absolute_positions = vec![-1i64; token_capacity];
        let mut source_span_starts = vec![-1i64; token_capacity];
        let mut source_span_ends = vec![-1i64; token_capacity];
        let mut token_mask = vec![false; token_capacity];
        for slot in 0..active_remote_tokens {
            token_state_values[slot * 4] = 0.2 + (slot as f32 * 0.01);
            token_state_values[slot * 4 + 1] = 0.1;
            token_state_values[slot * 4 + 2] = 0.0;
            token_state_values[slot * 4 + 3] = 0.3;
            token_mask[slot] = true;
        }
        if active_remote_tokens > 0 {
            let active_span_count = active_remote_tokens / PHASE1_LEAF_SIZE;
            for span_slot in 0..active_span_count {
                let span_start = (span_slot * PHASE1_LEAF_SIZE) as i64;
                let span_end = span_start + PHASE1_LEAF_SIZE as i64;
                for token_offset in 0..PHASE1_LEAF_SIZE {
                    let slot = span_slot * PHASE1_LEAF_SIZE + token_offset;
                    absolute_positions[slot] = span_start + token_offset as i64;
                    source_span_starts[slot] = span_start;
                    source_span_ends[slot] = span_end;
                }
            }
        }
        GatheredRetrievalContext::from_tensors(
            provenance,
            sealed_span_layout(),
            Tensor::<TestBackend, 3>::from_data(
                TensorData::new(token_state_values, [1, token_capacity, 4]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(absolute_positions, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(source_span_starts, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(source_span_ends, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(token_mask, [1, token_capacity]),
                &device,
            ),
        )
        .unwrap()
    }

    fn oracle_subset_context(active_remote_tokens: usize) -> GatheredRetrievalContext<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        let token_capacity = PHASE1_REMOTE_TOKEN_BUDGET;
        let mut token_state_values = vec![0.0f32; token_capacity * 4];
        let mut absolute_positions = vec![-1i64; token_capacity];
        let mut source_span_starts = vec![-1i64; token_capacity];
        let mut source_span_ends = vec![-1i64; token_capacity];
        let mut token_mask = vec![false; token_capacity];
        for slot in 0..active_remote_tokens {
            token_state_values[slot * 4] = 0.4 + (slot as f32 * 0.01);
            token_state_values[slot * 4 + 1] = 0.2;
            token_state_values[slot * 4 + 2] = 0.1;
            token_state_values[slot * 4 + 3] = 0.3;
            absolute_positions[slot] = slot as i64;
            source_span_starts[slot] = 0;
            source_span_ends[slot] = PHASE1_LEAF_SIZE as i64;
            token_mask[slot] = true;
        }
        GatheredRetrievalContext::from_tensors(
            GatheredRetrievalProvenance::OracleExactTokenSubset,
            exact_subset_layout(),
            Tensor::<TestBackend, 3>::from_data(
                TensorData::new(token_state_values, [1, token_capacity, 4]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(absolute_positions, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(source_span_starts, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(source_span_ends, [1, token_capacity]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(token_mask, [1, token_capacity]),
                &device,
            ),
        )
        .unwrap()
    }

    fn attention_input(
        mode: HybridRescuePrevalidationMode,
        query_position: i64,
        local_positions: [i64; 3],
        gathered: GatheredRetrievalContext<TestBackend>,
    ) -> RescueAttentionInput<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        RescueAttentionInput::new(
            mode,
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(vec![0.5f32, 0.1, 0.0, 0.2], [1, 4]),
                &device,
            ),
            Tensor::<TestBackend, 1, Int>::from_data(
                TensorData::new(vec![query_position], [1]),
                &device,
            ),
            Tensor::<TestBackend, 3>::from_data(
                TensorData::new(
                    vec![
                        0.0f32, 0.1, 0.0, 0.1, //
                        0.1, 0.0, 0.2, 0.0, //
                        0.3, 0.1, 0.2, 0.4,
                    ],
                    [1, 3, 4],
                ),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(local_positions.to_vec(), [1, 3]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(vec![true, true, true], [1, 3]),
                &device,
            ),
            gathered,
        )
        .unwrap()
    }

    #[test]
    fn baseline_rescue_attention_config_rejects_budget_mismatch() {
        let error = BaselineRescueAttentionConfig {
            token_state_dim: 4,
            attention_dim: 3,
            local_window_size: PHASE1_LOCAL_WINDOW_SIZE,
            routed_span_count: PHASE1_ROUTED_SPAN_COUNT,
            leaf_size: PHASE1_LEAF_SIZE,
            sink_token_count: 0,
            total_token_budget: PHASE1_TOTAL_TOKEN_BUDGET - 1,
            initializer: Initializer::Constant { value: 0.25 },
        }
        .validate()
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("rescue_attention.total_token_budget"))
        );
    }

    #[test]
    fn baseline_rescue_attention_rejects_unsealed_remote_spans() {
        let block = baseline_block();
        let error = block
            .attend(attention_input(
                HybridRescuePrevalidationMode::RoutedRemote,
                15,
                [13, 14, 15],
                gathered_context(GatheredRetrievalProvenance::Routed, PHASE1_LEAF_SIZE),
            ))
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("remote span end 16 must be less than or equal"))
        );
    }

    #[test]
    fn baseline_rescue_attention_returns_updated_state_and_attention_weights() {
        let block = baseline_block();
        let output = block
            .attend(attention_input(
                HybridRescuePrevalidationMode::RoutedRemote,
                17,
                [15, 16, 17],
                gathered_context(GatheredRetrievalProvenance::Routed, PHASE1_LEAF_SIZE),
            ))
            .unwrap();

        assert_eq!(output.updated_state().dims(), [1, 4]);
        assert_eq!(output.attention_weights().dims(), [1, 131]);
        assert!(output.diagnostics().mean_local_attention_mass > 0.0);
        assert!(output.diagnostics().mean_remote_attention_mass > 0.0);
    }

    #[test]
    fn baseline_rescue_attention_accepts_local_only_mode_with_masked_remote_capacity() {
        let block = baseline_block();
        let output = block
            .attend(attention_input(
                HybridRescuePrevalidationMode::LocalOnly,
                2,
                [0, 1, 2],
                gathered_context(GatheredRetrievalProvenance::Suppressed, 0),
            ))
            .unwrap();

        assert_eq!(output.updated_state().dims(), [1, 4]);
        assert!(output.diagnostics().mean_remote_attention_mass.abs() < f32::EPSILON);
    }

    #[test]
    fn baseline_rescue_attention_rejects_local_tokens_outside_visible_window() {
        let block = baseline_block();
        let error = block
            .attend(attention_input(
                HybridRescuePrevalidationMode::RoutedRemote,
                400,
                [1, 2, 400],
                gathered_context(GatheredRetrievalProvenance::Routed, PHASE1_LEAF_SIZE),
            ))
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("must lie within the last 256 visible local tokens"))
        );
    }

    #[test]
    fn baseline_rescue_attention_accepts_oracle_exact_subset_layout() {
        let block = baseline_block();
        let output = block
            .attend(attention_input(
                HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset,
                17,
                [15, 16, 17],
                oracle_subset_context(4),
            ))
            .unwrap();

        assert_eq!(output.updated_state().dims(), [1, 4]);
    }
}
