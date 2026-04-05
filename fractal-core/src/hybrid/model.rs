use burn::{
    module::Module,
    tensor::{backend::Backend, Bool, Int, Tensor, TensorData},
};

use crate::{
    error::FractalError,
    language_model_head::LanguageModelHead,
    v2::{
        ExactLeafRead, FractalRouterHead, FractalV2MemoryMode, FractalV2Model,
        FractalV2RetrievalTrace, LeafSummarizer, LocalTrunk, ReadFusion, TokenSpan, TreeMergeCell,
    },
};

use super::{
    rescue_attention::{
        RescueAttentionBlock, RescueAttentionDiagnostics, RescueAttentionInput,
        RescueAttentionShape, PHASE1_LOCAL_WINDOW_SIZE, PHASE1_TOTAL_TOKEN_BUDGET,
    },
    retrieval_gather::{GatheredCandidateRecall, SealedTokenStateStore},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridRescuePrevalidationMode {
    LocalOnly,
    RoutedRemote,
    OracleRemote,
    OracleRemoteWithOracleExactTokenSubset,
}

#[derive(Module, Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridModelShape {
    pub vocab_size: usize,
    pub token_state_dim: usize,
    pub rescue_attention: RescueAttentionShape,
}

#[derive(Debug, Clone)]
pub struct HybridRescueStepOutput<B: Backend> {
    query_position: usize,
    attention_weights: Tensor<B, 2>,
    attention_diagnostics: RescueAttentionDiagnostics,
    candidate_recall: Option<Vec<GatheredCandidateRecall>>,
}

#[derive(Debug, Clone)]
pub struct PreparedHybridRescueStep<B: Backend> {
    query_position: usize,
    input: RescueAttentionInput<B>,
    candidate_recall: Option<Vec<GatheredCandidateRecall>>,
}

#[derive(Debug, Clone)]
pub struct HybridRescueForwardOutput<B: Backend> {
    updated_token_states: Tensor<B, 3>,
    logits: Tensor<B, 3>,
    steps: Vec<HybridRescueStepOutput<B>>,
}

#[derive(Debug, Clone)]
struct LocalWindowContext<B: Backend> {
    token_states: Tensor<B, 3>,
    positions: Tensor<B, 2, Int>,
    mask: Tensor<B, 2, Bool>,
}

#[derive(Module, Debug)]
pub struct FractalHybridRescuePrevalidationModel<
    B: Backend,
    LT: Module<B>,
    LS: Module<B>,
    TM: Module<B>,
    RH: Module<B>,
    ER: Module<B>,
    RF: Module<B>,
    RA: Module<B>,
> {
    backbone: FractalV2Model<B, LT, LS, TM, RH, ER, RF>,
    rescue_attention: RA,
    shape: HybridModelShape,
}

impl HybridModelShape {
    pub fn validate(self) -> Result<Self, FractalError> {
        ensure_nonzero("hybrid_model.vocab_size", self.vocab_size)?;
        ensure_nonzero("hybrid_model.token_state_dim", self.token_state_dim)?;
        ensure_match(
            "hybrid_model.rescue_attention.token_state_dim",
            self.rescue_attention.token_state_dim,
            self.token_state_dim,
        )?;
        self.rescue_attention.validate()?;
        Ok(self)
    }
}

impl<B: Backend> HybridRescueStepOutput<B> {
    pub fn query_position(&self) -> usize {
        self.query_position
    }

    pub fn attention_weights(&self) -> Tensor<B, 2> {
        self.attention_weights.clone()
    }

    pub fn attention_diagnostics(&self) -> &RescueAttentionDiagnostics {
        &self.attention_diagnostics
    }

    pub fn candidate_recall(&self) -> Option<&[GatheredCandidateRecall]> {
        self.candidate_recall.as_deref()
    }
}

impl<B: Backend> PreparedHybridRescueStep<B> {
    pub fn query_position(&self) -> usize {
        self.query_position
    }

    pub fn input(&self) -> RescueAttentionInput<B> {
        self.input.clone()
    }

    pub fn candidate_recall(&self) -> Option<&[GatheredCandidateRecall]> {
        self.candidate_recall.as_deref()
    }
}

impl<B: Backend> HybridRescueForwardOutput<B> {
    pub fn updated_token_states(&self) -> Tensor<B, 3> {
        self.updated_token_states.clone()
    }

    pub fn logits(&self) -> Tensor<B, 3> {
        self.logits.clone()
    }

    pub fn steps(&self) -> &[HybridRescueStepOutput<B>] {
        &self.steps
    }
}

impl<B, LT, LS, TM, RH, ER, RF, RA>
    FractalHybridRescuePrevalidationModel<B, LT, LS, TM, RH, ER, RF, RA>
where
    B: Backend,
    LT: LocalTrunk<B> + Module<B>,
    LS: LeafSummarizer<B> + Module<B>,
    TM: TreeMergeCell<B> + Module<B>,
    RH: FractalRouterHead<B> + Module<B>,
    ER: ExactLeafRead<B> + Module<B>,
    RF: ReadFusion<B> + Module<B>,
    RA: RescueAttentionBlock<B> + Module<B>,
{
    pub fn new(
        backbone: FractalV2Model<B, LT, LS, TM, RH, ER, RF>,
        rescue_attention: RA,
    ) -> Result<Self, FractalError> {
        let backbone_shape = backbone.shape();
        let output_dims = backbone.output().logical_dims();
        let shape = HybridModelShape {
            vocab_size: backbone_shape.vocab_size,
            token_state_dim: backbone_shape.local_trunk.root_readout_dim,
            rescue_attention: rescue_attention.shape(),
        }
        .validate()?;
        ensure_match(
            "hybrid_model.backbone.local_trunk.root_count",
            backbone_shape.local_trunk.root_count,
            1,
        )?;
        ensure_match(
            "hybrid_model.backbone.router.query_dim",
            backbone_shape.router.query_dim,
            shape.token_state_dim,
        )?;
        ensure_match(
            "hybrid_model.backbone.output.readout_width",
            output_dims[0],
            shape.token_state_dim,
        )?;
        ensure_match(
            "hybrid_model.backbone.local_trunk.leaf_size",
            backbone_shape.local_trunk.leaf_size,
            shape.rescue_attention.leaf_size,
        )?;
        let router_selection_capacity = backbone_shape
            .router
            .head_count
            .checked_mul(backbone_shape.router.top_leaf_reads)
            .ok_or_else(|| {
                FractalError::InvalidConfig(
                    "hybrid_model router selection capacity overflowed".to_string(),
                )
            })?;
        if router_selection_capacity < shape.rescue_attention.routed_span_count {
            return Err(FractalError::InvalidConfig(format!(
                "hybrid_model router selection capacity {} must cover rescue routed span count {}",
                router_selection_capacity, shape.rescue_attention.routed_span_count
            )));
        }

        Ok(Self {
            backbone,
            rescue_attention,
            shape,
        })
    }

    pub fn shape(&self) -> HybridModelShape {
        self.shape
    }

    pub fn backbone(&self) -> &FractalV2Model<B, LT, LS, TM, RH, ER, RF> {
        &self.backbone
    }

    pub fn rescue_attention(&self) -> &RA {
        &self.rescue_attention
    }

    pub fn output(&self) -> &LanguageModelHead<B> {
        self.backbone.output()
    }

    pub fn forward_with_mode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mode: HybridRescuePrevalidationMode,
    ) -> Result<HybridRescueForwardOutput<B>, FractalError> {
        self.forward_with_mode_and_oracle_spans(input_ids, mode, None)
    }

    pub fn forward_with_mode_and_oracle_spans(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mode: HybridRescuePrevalidationMode,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<HybridRescueForwardOutput<B>, FractalError> {
        validate_oracle_request(input_ids.dims(), mode, oracle_evidence_spans)?;
        let trace = self
            .backbone
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)?;
        self.forward_from_retrieval_trace(trace, mode, oracle_evidence_spans)
    }

    pub fn prepare_rescue_steps_with_mode_and_oracle_spans(
        &self,
        input_ids: Tensor<B, 2, Int>,
        mode: HybridRescuePrevalidationMode,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<Vec<PreparedHybridRescueStep<B>>, FractalError> {
        validate_oracle_request(input_ids.dims(), mode, oracle_evidence_spans)?;
        let trace = self
            .backbone
            .forward_retrieval_trace_with_memory_mode(input_ids, FractalV2MemoryMode::TreeOnly)?;
        self.prepare_rescue_steps_from_retrieval_trace(trace, mode, oracle_evidence_spans)
    }

    pub fn prepare_rescue_steps_from_retrieval_trace(
        &self,
        trace: FractalV2RetrievalTrace<B>,
        mode: HybridRescuePrevalidationMode,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<Vec<PreparedHybridRescueStep<B>>, FractalError> {
        let steps = trace.steps();
        ensure_nonzero("hybrid_forward.trace_step_count", steps.len())?;
        let batch_size = steps[0].root_readouts().dims()[0];
        let seq_len = steps.len();
        validate_oracle_request([batch_size, seq_len], mode, oracle_evidence_spans)?;
        self.prepare_steps_from_trace_internal(trace, mode, oracle_evidence_spans)
    }

    pub fn forward_from_retrieval_trace(
        &self,
        trace: FractalV2RetrievalTrace<B>,
        mode: HybridRescuePrevalidationMode,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<HybridRescueForwardOutput<B>, FractalError> {
        let prepared_steps =
            self.prepare_rescue_steps_from_retrieval_trace(trace, mode, oracle_evidence_spans)?;
        let batch_size = prepared_steps[0].input().query_state().dims()[0];
        let mut updated_states = Vec::with_capacity(prepared_steps.len());
        let mut logits = Vec::with_capacity(prepared_steps.len());
        let mut step_outputs = Vec::with_capacity(prepared_steps.len());

        for prepared_step in prepared_steps {
            let rescue_output = self.rescue_attention.attend(prepared_step.input())?;
            let updated_state = rescue_output.updated_state();
            let step_logits = self.output().forward(updated_state.clone());
            updated_states.push(updated_state.clone().reshape([
                batch_size,
                1,
                self.shape.token_state_dim,
            ]));
            logits.push(
                step_logits
                    .clone()
                    .reshape([batch_size, 1, self.shape.vocab_size]),
            );
            step_outputs.push(HybridRescueStepOutput {
                query_position: prepared_step.query_position,
                attention_weights: rescue_output.attention_weights(),
                attention_diagnostics: rescue_output.diagnostics().clone(),
                candidate_recall: prepared_step.candidate_recall,
            });
        }

        Ok(HybridRescueForwardOutput {
            updated_token_states: Tensor::cat(updated_states, 1),
            logits: Tensor::cat(logits, 1),
            steps: step_outputs,
        })
    }

    fn prepare_steps_from_trace_internal(
        &self,
        trace: FractalV2RetrievalTrace<B>,
        mode: HybridRescuePrevalidationMode,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<Vec<PreparedHybridRescueStep<B>>, FractalError> {
        let steps = trace.steps();
        ensure_nonzero("hybrid_forward.trace_step_count", steps.len())?;
        let batch_size = steps[0].root_readouts().dims()[0];
        let token_state_dim = self.shape.token_state_dim;
        let device = steps[0].root_readouts().device();
        let mut sealed_store = SealedTokenStateStore::new(
            batch_size,
            self.shape.rescue_attention.leaf_size,
            token_state_dim,
        )?;
        let mut current_leaf = Vec::with_capacity(self.shape.rescue_attention.leaf_size);
        let mut token_history = Vec::with_capacity(steps.len());
        let mut prepared_steps = Vec::with_capacity(steps.len());

        for (position, step) in steps.iter().enumerate() {
            let query_position = position + 1;
            let token_state =
                single_root_token_state(step.root_readouts(), self.shape.token_state_dim)?;
            token_history.push(token_state.clone());
            current_leaf.push(token_state.clone());

            if let Some(sealed_leaf) = step.sealed_leaf() {
                ensure_match(
                    "hybrid_forward.current_leaf_len_on_seal",
                    current_leaf.len(),
                    self.shape.rescue_attention.leaf_size,
                )?;
                let sealed_token_states = Tensor::cat(
                    current_leaf
                        .iter()
                        .map(|token_state| {
                            token_state
                                .clone()
                                .reshape([batch_size, 1, self.shape.token_state_dim])
                        })
                        .collect(),
                    1,
                );
                sealed_store.push_sealed_leaf(sealed_leaf.shared_span(), sealed_token_states)?;
                current_leaf.clear();
            }

            let local_window = build_local_window(
                batch_size,
                self.shape.token_state_dim,
                mode,
                &token_history,
                &device,
            )?;
            let oracle_for_step = oracle_evidence_spans.map(|spans| vec![spans[position]]);
            let (gathered_remote, candidate_recall) = sealed_store.gather_for_mode(
                mode,
                step.routed(),
                query_position,
                oracle_for_step.as_deref(),
            )?;
            let rescue_input = RescueAttentionInput::new(
                mode,
                token_state,
                Tensor::<B, 1, Int>::from_data(
                    TensorData::new(vec![query_position as i64; batch_size], [batch_size]),
                    &device,
                ),
                local_window.token_states,
                local_window.positions,
                local_window.mask,
                gathered_remote,
            )?;
            prepared_steps.push(PreparedHybridRescueStep {
                query_position,
                input: rescue_input,
                candidate_recall,
            });
        }

        Ok(prepared_steps)
    }
}

fn validate_oracle_request(
    input_dims: [usize; 2],
    mode: HybridRescuePrevalidationMode,
    oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
) -> Result<(), FractalError> {
    let [batch_size, seq_len] = input_dims;
    ensure_nonzero("hybrid_forward.batch_size", batch_size)?;
    ensure_nonzero("hybrid_forward.seq_len", seq_len)?;
    if let Some(spans) = oracle_evidence_spans {
        ensure_match(
            "hybrid_forward.oracle_evidence_spans.len",
            spans.len(),
            seq_len,
        )?;
        ensure_match(
            "hybrid_forward.oracle_evidence_spans.batch_size",
            batch_size,
            1,
        )?;
    } else if matches!(
        mode,
        HybridRescuePrevalidationMode::OracleRemote
            | HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset
    ) {
        return Err(FractalError::InvalidConfig(
            "hybrid_forward oracle modes require oracle_evidence_spans".to_string(),
        ));
    }

    Ok(())
}

fn build_local_window<B: Backend>(
    batch_size: usize,
    token_state_dim: usize,
    mode: HybridRescuePrevalidationMode,
    token_history: &[Tensor<B, 2>],
    device: &B::Device,
) -> Result<LocalWindowContext<B>, FractalError> {
    let local_token_budget = match mode {
        HybridRescuePrevalidationMode::LocalOnly => PHASE1_TOTAL_TOKEN_BUDGET,
        HybridRescuePrevalidationMode::RoutedRemote
        | HybridRescuePrevalidationMode::OracleRemote
        | HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset => {
            PHASE1_LOCAL_WINDOW_SIZE
        }
    };
    let start_index = token_history.len().saturating_sub(local_token_budget);
    let active_count = token_history.len() - start_index;
    ensure_nonzero("hybrid_forward.local_window.active_count", active_count)?;
    let local_token_states = Tensor::cat(
        token_history[start_index..]
            .iter()
            .map(|token_state| {
                token_state
                    .clone()
                    .reshape([batch_size, 1, token_state_dim])
            })
            .collect(),
        1,
    );
    let mut position_data = Vec::with_capacity(batch_size * active_count);
    let mut mask_data = Vec::with_capacity(batch_size * active_count);
    for _ in 0..batch_size {
        for absolute_position in start_index..token_history.len() {
            position_data.push(absolute_position as i64);
            mask_data.push(true);
        }
    }

    Ok(LocalWindowContext {
        token_states: local_token_states,
        positions: Tensor::<B, 2, Int>::from_data(
            TensorData::new(position_data, [batch_size, active_count]),
            device,
        ),
        mask: Tensor::<B, 2, Bool>::from_data(
            TensorData::new(mask_data, [batch_size, active_count]),
            device,
        ),
    })
}

fn single_root_token_state<B: Backend>(
    root_readouts: Tensor<B, 3>,
    token_state_dim: usize,
) -> Result<Tensor<B, 2>, FractalError> {
    let [batch_size, root_count, readout_dim] = root_readouts.dims();
    ensure_match("hybrid_forward.root_readouts.root_count", root_count, 1)?;
    ensure_match(
        "hybrid_forward.root_readouts.readout_dim",
        readout_dim,
        token_state_dim,
    )?;
    Ok(root_readouts
        .narrow(1, 0, 1)
        .reshape([batch_size, token_state_dim]))
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

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        nn::Initializer,
        tensor::{Tensor, TensorData},
    };

    use crate::v2::{
        BaselineExactLeafRead, BaselineExactLeafReadConfig, BaselineFractalRouterHead,
        BaselineFractalRouterHeadConfig, BaselineLeafSummarizer, BaselineLeafSummarizerConfig,
        BaselineLocalTrunk, BaselineLocalTrunkConfig, BaselineReadFusion, BaselineReadFusionConfig,
        BaselineTreeMergeCell, BaselineTreeMergeCellConfig, FractalV2Components,
    };

    use super::*;
    use crate::hybrid::BaselineRescueAttentionConfig;

    type TestBackend = Candle<f32, i64>;
    type TestBackbone = FractalV2Model<
        TestBackend,
        BaselineLocalTrunk<TestBackend>,
        BaselineLeafSummarizer<TestBackend>,
        BaselineTreeMergeCell<TestBackend>,
        BaselineFractalRouterHead<TestBackend>,
        BaselineExactLeafRead<TestBackend>,
        BaselineReadFusion<TestBackend>,
    >;
    type TestHybrid = FractalHybridRescuePrevalidationModel<
        TestBackend,
        BaselineLocalTrunk<TestBackend>,
        BaselineLeafSummarizer<TestBackend>,
        BaselineTreeMergeCell<TestBackend>,
        BaselineFractalRouterHead<TestBackend>,
        BaselineExactLeafRead<TestBackend>,
        BaselineReadFusion<TestBackend>,
        crate::hybrid::BaselineRescueAttentionBlock<TestBackend>,
    >;

    fn test_backbone(device: &<TestBackend as Backend>::Device) -> TestBackbone {
        FractalV2Model::new(
            64,
            8,
            FractalV2Components {
                local_trunk: BaselineLocalTrunkConfig::new(8, 1, 6, 8, 16)
                    .try_init(device)
                    .unwrap(),
                leaf_summarizer: BaselineLeafSummarizerConfig {
                    readout_dim: 8,
                    leaf_size: 16,
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    token_cache_key_dim: 4,
                    token_cache_value_dim: 6,
                }
                .try_init(device)
                .unwrap(),
                tree_merge_cell: BaselineTreeMergeCellConfig {
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    scale_embedding_dim: 4,
                }
                .try_init(device)
                .unwrap(),
                router: BaselineFractalRouterHeadConfig {
                    query_dim: 8,
                    key_dim: 4,
                    head_count: 1,
                    beam_width: 8,
                    top_leaf_reads: 8,
                    allow_early_stop: false,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
                exact_read: BaselineExactLeafReadConfig {
                    query_dim: 8,
                    key_dim: 4,
                    value_dim: 6,
                    head_count: 1,
                    top_leaf_reads: 8,
                    leaf_size: 16,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
                read_fusion: BaselineReadFusionConfig {
                    root_count: 1,
                    root_readout_dim: 8,
                    routed_value_dim: 5,
                    exact_read_value_dim: 6,
                    fused_readout_dim: 8,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(device)
                .unwrap(),
            },
            device,
        )
        .unwrap()
    }

    fn test_model(device: &<TestBackend as Backend>::Device) -> TestHybrid {
        let backbone = test_backbone(device);
        let rescue_attention = BaselineRescueAttentionConfig {
            token_state_dim: 8,
            attention_dim: 4,
            local_window_size: PHASE1_LOCAL_WINDOW_SIZE,
            routed_span_count: 8,
            leaf_size: 16,
            sink_token_count: 0,
            total_token_budget: PHASE1_TOTAL_TOKEN_BUDGET,
            initializer: Initializer::Constant { value: 0.25 },
        }
        .init::<TestBackend>(device);
        FractalHybridRescuePrevalidationModel::new(backbone, rescue_attention).unwrap()
    }

    #[test]
    fn hybrid_model_rejects_multi_root_backbone() {
        let device = <TestBackend as Backend>::Device::default();
        let backbone = FractalV2Model::new(
            64,
            8,
            FractalV2Components {
                local_trunk: BaselineLocalTrunkConfig::new(8, 2, 6, 4, 16)
                    .try_init(&device)
                    .unwrap(),
                leaf_summarizer: BaselineLeafSummarizerConfig {
                    readout_dim: 4,
                    leaf_size: 16,
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    token_cache_key_dim: 4,
                    token_cache_value_dim: 6,
                }
                .try_init(&device)
                .unwrap(),
                tree_merge_cell: BaselineTreeMergeCellConfig {
                    summary_dim: 6,
                    key_dim: 4,
                    value_dim: 5,
                    scale_embedding_dim: 4,
                }
                .try_init(&device)
                .unwrap(),
                router: BaselineFractalRouterHeadConfig {
                    query_dim: 4,
                    key_dim: 4,
                    head_count: 4,
                    beam_width: 2,
                    top_leaf_reads: 2,
                    allow_early_stop: false,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(&device)
                .unwrap(),
                exact_read: BaselineExactLeafReadConfig {
                    query_dim: 4,
                    key_dim: 4,
                    value_dim: 6,
                    head_count: 4,
                    top_leaf_reads: 2,
                    leaf_size: 16,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(&device)
                .unwrap(),
                read_fusion: BaselineReadFusionConfig {
                    root_count: 2,
                    root_readout_dim: 4,
                    routed_value_dim: 5,
                    exact_read_value_dim: 6,
                    fused_readout_dim: 8,
                    initializer: Initializer::Uniform {
                        min: -0.08,
                        max: 0.08,
                    },
                }
                .try_init(&device)
                .unwrap(),
            },
            &device,
        )
        .unwrap();
        let rescue_attention = BaselineRescueAttentionConfig {
            token_state_dim: 4,
            attention_dim: 4,
            local_window_size: PHASE1_LOCAL_WINDOW_SIZE,
            routed_span_count: 8,
            leaf_size: 16,
            sink_token_count: 0,
            total_token_budget: PHASE1_TOTAL_TOKEN_BUDGET,
            initializer: Initializer::Constant { value: 0.25 },
        }
        .init::<TestBackend>(&device);

        let error =
            FractalHybridRescuePrevalidationModel::new(backbone, rescue_attention).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("root_count"))
        );
    }

    #[test]
    fn hybrid_model_forward_local_only_runs() {
        let device = <TestBackend as Backend>::Device::default();
        let model = test_model(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                (0..20).map(|index| (index % 64) as i64).collect::<Vec<_>>(),
                [1, 20],
            ),
            &device,
        );

        let output = model
            .forward_with_mode(input_ids, HybridRescuePrevalidationMode::LocalOnly)
            .unwrap();

        assert_eq!(output.updated_token_states().dims(), [1, 20, 8]);
        assert_eq!(output.logits().dims(), [1, 20, 64]);
        assert_eq!(output.steps().len(), 20);
    }

    #[test]
    fn hybrid_model_forward_oracle_exact_subset_tracks_candidate_recall() {
        let device = <TestBackend as Backend>::Device::default();
        let model = test_model(&device);
        let input_ids = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::new(
                (0..20).map(|index| (index % 64) as i64).collect::<Vec<_>>(),
                [1, 20],
            ),
            &device,
        );
        let mut oracle_spans = vec![None; 20];
        for span in oracle_spans.iter_mut().skip(16) {
            *span = Some(TokenSpan::new(0, 1).unwrap());
        }

        let output = model
            .forward_with_mode_and_oracle_spans(
                input_ids,
                HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset,
                Some(&oracle_spans),
            )
            .unwrap();

        let final_recall = output.steps().last().unwrap().candidate_recall().unwrap();
        assert_eq!(final_recall.len(), 1);
        assert!(final_recall[0].evidence_span_recalled);
        assert_eq!(final_recall[0].gathered_evidence_token_count, 1);
        assert_eq!(final_recall[0].evidence_token_recall(), 1.0);
    }
}
