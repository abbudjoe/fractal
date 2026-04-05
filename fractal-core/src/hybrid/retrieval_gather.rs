use std::{cmp::Ordering, collections::{BTreeMap, BTreeSet}};

use burn::tensor::{backend::Backend, Bool, Int, Tensor, TensorData};

use crate::{
    error::FractalError,
    v2::{FractalRouteOutput, TokenSpan},
};

use super::model::HybridRescuePrevalidationMode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatheredRetrievalLayout {
    SealedSpanPacks {
        max_span_count: usize,
        leaf_size: usize,
    },
    ExactTokenSubset {
        max_token_count: usize,
    },
}

impl GatheredRetrievalLayout {
    pub fn validate(self) -> Result<Self, FractalError> {
        match self {
            Self::SealedSpanPacks {
                max_span_count,
                leaf_size,
            } => {
                ensure_nonzero(
                    "gathered_retrieval.layout.sealed_span_packs.max_span_count",
                    max_span_count,
                )?;
                ensure_nonzero("gathered_retrieval.layout.sealed_span_packs.leaf_size", leaf_size)?;
            }
            Self::ExactTokenSubset { max_token_count } => {
                ensure_nonzero(
                    "gathered_retrieval.layout.exact_token_subset.max_token_count",
                    max_token_count,
                )?;
            }
        }

        Ok(self)
    }

    pub fn token_capacity(self) -> usize {
        match self {
            Self::SealedSpanPacks {
                max_span_count,
                leaf_size,
            } => max_span_count * leaf_size,
            Self::ExactTokenSubset { max_token_count } => max_token_count,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatheredRetrievalProvenance {
    Suppressed,
    Routed,
    Oracle,
    OracleExactTokenSubset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatheredRetrievalContextShape {
    pub batch_size: usize,
    pub token_capacity: usize,
    pub token_state_dim: usize,
    pub layout: GatheredRetrievalLayout,
    pub provenance: GatheredRetrievalProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatheredCandidateRecall {
    pub evidence_span_recalled: bool,
    pub evidence_token_count: usize,
    pub gathered_evidence_token_count: usize,
}

impl GatheredCandidateRecall {
    pub fn evidence_token_recall(&self) -> f32 {
        if self.evidence_token_count == 0 {
            0.0
        } else {
            self.gathered_evidence_token_count as f32 / self.evidence_token_count as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatheredRetrievalContext<B: Backend> {
    provenance: GatheredRetrievalProvenance,
    layout: GatheredRetrievalLayout,
    token_states: Tensor<B, 3>,
    absolute_positions: Tensor<B, 2, Int>,
    source_span_starts: Tensor<B, 2, Int>,
    source_span_ends: Tensor<B, 2, Int>,
    token_mask: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone)]
pub struct SealedTokenStateStore<B: Backend> {
    batch_size: usize,
    leaf_size: usize,
    token_state_dim: usize,
    shared_spans: Vec<TokenSpan>,
    token_states: Vec<Tensor<B, 3>>,
}

#[derive(Debug, Clone)]
struct GatheredRowBatch<B: Backend> {
    token_states: Tensor<B, 3>,
    absolute_positions: Vec<i64>,
    source_span_starts: Vec<i64>,
    source_span_ends: Vec<i64>,
    token_mask: Vec<bool>,
}

impl<B: Backend> GatheredRetrievalContext<B> {
    pub fn from_tensors(
        provenance: GatheredRetrievalProvenance,
        layout: GatheredRetrievalLayout,
        token_states: Tensor<B, 3>,
        absolute_positions: Tensor<B, 2, Int>,
        source_span_starts: Tensor<B, 2, Int>,
        source_span_ends: Tensor<B, 2, Int>,
        token_mask: Tensor<B, 2, Bool>,
    ) -> Result<Self, FractalError> {
        let layout = layout.validate()?;
        let [batch_size, token_capacity, token_state_dim] = token_states.dims();
        ensure_nonzero("gathered_retrieval.batch_size", batch_size)?;
        ensure_nonzero("gathered_retrieval.token_state_dim", token_state_dim)?;
        ensure_match(
            "gathered_retrieval.token_capacity",
            token_capacity,
            layout.token_capacity(),
        )?;
        ensure_dims2(
            "gathered_retrieval.absolute_positions",
            absolute_positions.dims(),
            [batch_size, token_capacity],
        )?;
        ensure_dims2(
            "gathered_retrieval.source_span_starts",
            source_span_starts.dims(),
            [batch_size, token_capacity],
        )?;
        ensure_dims2(
            "gathered_retrieval.source_span_ends",
            source_span_ends.dims(),
            [batch_size, token_capacity],
        )?;
        ensure_dims2(
            "gathered_retrieval.token_mask",
            token_mask.dims(),
            [batch_size, token_capacity],
        )?;

        validate_position_contract(
            layout,
            batch_size,
            token_capacity,
            absolute_positions.clone(),
            source_span_starts.clone(),
            source_span_ends.clone(),
            token_mask.clone(),
        )?;

        Ok(Self {
            provenance,
            layout,
            token_states,
            absolute_positions,
            source_span_starts,
            source_span_ends,
            token_mask,
        })
    }

    pub fn shape(&self) -> GatheredRetrievalContextShape {
        let [batch_size, token_capacity, token_state_dim] = self.token_states.dims();
        GatheredRetrievalContextShape {
            batch_size,
            token_capacity,
            token_state_dim,
            layout: self.layout,
            provenance: self.provenance,
        }
    }

    pub fn token_states(&self) -> Tensor<B, 3> {
        self.token_states.clone()
    }

    pub fn absolute_positions(&self) -> Tensor<B, 2, Int> {
        self.absolute_positions.clone()
    }

    pub fn source_span_starts(&self) -> Tensor<B, 2, Int> {
        self.source_span_starts.clone()
    }

    pub fn source_span_ends(&self) -> Tensor<B, 2, Int> {
        self.source_span_ends.clone()
    }

    pub fn token_mask(&self) -> Tensor<B, 2, Bool> {
        self.token_mask.clone()
    }

    pub fn active_token_counts(&self) -> Result<Vec<usize>, FractalError> {
        let shape = self.shape();
        let token_mask =
            tensor_data_to_bool(self.token_mask.clone(), "gathered_retrieval.token_mask")?;
        let mut counts = Vec::with_capacity(shape.batch_size);
        for batch_index in 0..shape.batch_size {
            let mut active_count = 0usize;
            for slot in 0..shape.token_capacity {
                let flat_index = batch_index * shape.token_capacity + slot;
                if token_mask[flat_index] {
                    active_count += 1;
                }
            }
            counts.push(active_count);
        }
        Ok(counts)
    }

    pub fn candidate_recall_for_span(
        &self,
        evidence_span: TokenSpan,
    ) -> Result<Vec<GatheredCandidateRecall>, FractalError> {
        let shape = self.shape();
        let absolute_positions = tensor_data_to_i64(
            self.absolute_positions.clone(),
            "gathered_retrieval.absolute_positions",
        )?;
        let source_span_starts = tensor_data_to_i64(
            self.source_span_starts.clone(),
            "gathered_retrieval.source_span_starts",
        )?;
        let source_span_ends = tensor_data_to_i64(
            self.source_span_ends.clone(),
            "gathered_retrieval.source_span_ends",
        )?;
        let token_mask =
            tensor_data_to_bool(self.token_mask.clone(), "gathered_retrieval.token_mask")?;

        let mut recalls = Vec::with_capacity(shape.batch_size);
        for batch_index in 0..shape.batch_size {
            let mut evidence_span_recalled = false;
            let mut gathered_evidence_positions = BTreeSet::new();
            for slot in 0..shape.token_capacity {
                let flat_index = batch_index * shape.token_capacity + slot;
                if !token_mask[flat_index] {
                    continue;
                }
                let span_start = usize::try_from(source_span_starts[flat_index]).map_err(|_| {
                    FractalError::InvalidState(
                        "gathered retrieval source span start must be non-negative".to_string(),
                    )
                })?;
                let span_end = usize::try_from(source_span_ends[flat_index]).map_err(|_| {
                    FractalError::InvalidState(
                        "gathered retrieval source span end must be non-negative".to_string(),
                    )
                })?;
                let source_span = TokenSpan::new(span_start, span_end)?;
                if spans_overlap(source_span, evidence_span) {
                    evidence_span_recalled = true;
                }

                let absolute_position =
                    usize::try_from(absolute_positions[flat_index]).map_err(|_| {
                        FractalError::InvalidState(
                            "gathered retrieval absolute position must be non-negative".to_string(),
                        )
                    })?;
                if (evidence_span.start()..evidence_span.end()).contains(&absolute_position) {
                    gathered_evidence_positions.insert(absolute_position);
                }
            }

            recalls.push(GatheredCandidateRecall {
                evidence_span_recalled,
                evidence_token_count: evidence_span.len(),
                gathered_evidence_token_count: gathered_evidence_positions.len(),
            });
        }

        Ok(recalls)
    }
}

impl<B: Backend> SealedTokenStateStore<B> {
    pub fn new(
        batch_size: usize,
        leaf_size: usize,
        token_state_dim: usize,
    ) -> Result<Self, FractalError> {
        ensure_nonzero("sealed_token_state_store.batch_size", batch_size)?;
        ensure_nonzero("sealed_token_state_store.leaf_size", leaf_size)?;
        ensure_nonzero("sealed_token_state_store.token_state_dim", token_state_dim)?;

        Ok(Self {
            batch_size,
            leaf_size,
            token_state_dim,
            shared_spans: Vec::new(),
            token_states: Vec::new(),
        })
    }

    pub fn shared_spans(&self) -> &[TokenSpan] {
        &self.shared_spans
    }

    pub fn push_sealed_leaf(
        &mut self,
        shared_span: TokenSpan,
        token_states: Tensor<B, 3>,
    ) -> Result<usize, FractalError> {
        ensure_match(
            "sealed_token_state_store.push_sealed_leaf.shared_span.len",
            shared_span.len(),
            self.leaf_size,
        )?;
        let [batch_size, leaf_size, token_state_dim] = token_states.dims();
        ensure_match(
            "sealed_token_state_store.push_sealed_leaf.batch_size",
            batch_size,
            self.batch_size,
        )?;
        ensure_match(
            "sealed_token_state_store.push_sealed_leaf.leaf_size",
            leaf_size,
            self.leaf_size,
        )?;
        ensure_match(
            "sealed_token_state_store.push_sealed_leaf.token_state_dim",
            token_state_dim,
            self.token_state_dim,
        )?;
        if let Some(previous_span) = self.shared_spans.last().copied() {
            ensure_match(
                "sealed_token_state_store.push_sealed_leaf.shared_span.start",
                shared_span.start(),
                previous_span.end(),
            )?;
        } else if shared_span.start() != 0 {
            return Err(FractalError::InvalidConfig(format!(
                "sealed_token_state_store first shared span must start at 0, got {}",
                shared_span.start()
            )));
        }

        let leaf_index = self.shared_spans.len();
        self.shared_spans.push(shared_span);
        self.token_states.push(token_states);
        Ok(leaf_index)
    }

    pub fn gather_for_mode(
        &self,
        mode: HybridRescuePrevalidationMode,
        routed: &FractalRouteOutput<B>,
        query_position: usize,
        oracle_evidence_spans: Option<&[Option<TokenSpan>]>,
    ) -> Result<(GatheredRetrievalContext<B>, Option<Vec<GatheredCandidateRecall>>), FractalError> {
        let [batch_size, _, _] = routed.selected_leaf_indices().dims();
        ensure_match(
            "sealed_token_state_store.gather_for_mode.batch_size",
            batch_size,
            self.batch_size,
        )?;
        if let Some(spans) = oracle_evidence_spans {
            ensure_match(
                "sealed_token_state_store.gather_for_mode.oracle_batch_size",
                spans.len(),
                batch_size,
            )?;
        } else if matches!(
            mode,
            HybridRescuePrevalidationMode::OracleRemote
                | HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset
        ) {
            return Err(FractalError::InvalidConfig(
                "sealed_token_state_store.gather_for_mode oracle modes require oracle_evidence_spans"
                    .to_string(),
            ));
        }

        let (layout, provenance) = match mode {
            HybridRescuePrevalidationMode::LocalOnly => (
                GatheredRetrievalLayout::SealedSpanPacks {
                    max_span_count: 8,
                    leaf_size: self.leaf_size,
                },
                GatheredRetrievalProvenance::Suppressed,
            ),
            HybridRescuePrevalidationMode::RoutedRemote => (
                GatheredRetrievalLayout::SealedSpanPacks {
                    max_span_count: 8,
                    leaf_size: self.leaf_size,
                },
                GatheredRetrievalProvenance::Routed,
            ),
            HybridRescuePrevalidationMode::OracleRemote => (
                GatheredRetrievalLayout::SealedSpanPacks {
                    max_span_count: 8,
                    leaf_size: self.leaf_size,
                },
                GatheredRetrievalProvenance::Oracle,
            ),
            HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset => (
                GatheredRetrievalLayout::ExactTokenSubset {
                    max_token_count: 8 * self.leaf_size,
                },
                GatheredRetrievalProvenance::OracleExactTokenSubset,
            ),
        };
        let token_capacity = layout.token_capacity();
        let device = routed.selected_leaf_scores().device();

        let mut batch_token_rows = Vec::with_capacity(batch_size);
        let mut absolute_positions = Vec::with_capacity(batch_size * token_capacity);
        let mut source_span_starts = Vec::with_capacity(batch_size * token_capacity);
        let mut source_span_ends = Vec::with_capacity(batch_size * token_capacity);
        let mut token_mask = Vec::with_capacity(batch_size * token_capacity);

        for batch_index in 0..batch_size {
            let oracle_span = oracle_evidence_spans.and_then(|spans| spans[batch_index]);
            let row_batch = match mode {
                HybridRescuePrevalidationMode::LocalOnly => {
                    self.build_inactive_rows(token_capacity, &device)
                }
                HybridRescuePrevalidationMode::RoutedRemote => {
                    let selected_leaf_indices =
                        self.select_routed_leaf_indices(routed, batch_index, query_position, 8)?;
                    self.build_sealed_span_pack_rows(
                        batch_index,
                        &selected_leaf_indices,
                        token_capacity,
                        &device,
                    )?
                }
                HybridRescuePrevalidationMode::OracleRemote => {
                    if let Some(oracle_span) = oracle_span {
                        let selected_leaf_indices =
                            self.select_oracle_leaf_indices(oracle_span, query_position, 8);
                        self.build_sealed_span_pack_rows(
                            batch_index,
                            &selected_leaf_indices,
                            token_capacity,
                            &device,
                        )?
                    } else {
                        self.build_inactive_rows(token_capacity, &device)
                    }
                }
                HybridRescuePrevalidationMode::OracleRemoteWithOracleExactTokenSubset => {
                    if let Some(oracle_span) = oracle_span {
                        let selected_tokens = self.select_oracle_exact_tokens(
                            oracle_span,
                            query_position,
                            token_capacity,
                        )?;
                        self.build_exact_token_subset_rows(
                            batch_index,
                            &selected_tokens,
                            token_capacity,
                            &device,
                        )?
                    } else {
                        self.build_inactive_rows(token_capacity, &device)
                    }
                }
            };

            batch_token_rows.push(row_batch.token_states);
            absolute_positions.extend(row_batch.absolute_positions);
            source_span_starts.extend(row_batch.source_span_starts);
            source_span_ends.extend(row_batch.source_span_ends);
            token_mask.extend(row_batch.token_mask);
        }

        let token_states = Tensor::cat(batch_token_rows, 0);
        let context = GatheredRetrievalContext::from_tensors(
            provenance,
            layout,
            token_states,
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(absolute_positions, [batch_size, token_capacity]),
                &device,
            ),
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(source_span_starts, [batch_size, token_capacity]),
                &device,
            ),
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(source_span_ends, [batch_size, token_capacity]),
                &device,
            ),
            Tensor::<B, 2, Bool>::from_data(
                TensorData::new(token_mask, [batch_size, token_capacity]),
                &device,
            ),
        )?;

        let recalls = if let Some(spans) = oracle_evidence_spans {
            let mut per_batch = Vec::with_capacity(batch_size);
            for (batch_index, evidence_span) in spans.iter().copied().enumerate() {
                if let Some(evidence_span) = evidence_span {
                    let recall = context
                        .candidate_recall_for_span(evidence_span)?
                        .into_iter()
                        .nth(batch_index)
                        .ok_or_else(|| {
                            FractalError::InvalidState(format!(
                                "gathered retrieval candidate recall missing batch index {batch_index}"
                            ))
                        })?;
                    per_batch.push(recall);
                } else {
                    per_batch.push(GatheredCandidateRecall {
                        evidence_span_recalled: false,
                        evidence_token_count: 0,
                        gathered_evidence_token_count: 0,
                    });
                }
            }
            Some(per_batch)
        } else {
            None
        };

        Ok((context, recalls))
    }

    fn select_routed_leaf_indices(
        &self,
        routed: &FractalRouteOutput<B>,
        batch_index: usize,
        query_position: usize,
        max_span_count: usize,
    ) -> Result<Vec<usize>, FractalError> {
        let mut by_leaf = BTreeMap::<usize, (TokenSpan, f32)>::new();
        for head_trace in routed.traces() {
            let batch_route = head_trace.batch_routes.get(batch_index).ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "sealed_token_state_store batch route {batch_index} missing from routed trace"
                ))
            })?;
            for selection_index in 0..batch_route.selected_leaf_indices.len() {
                let leaf_index = batch_route.selected_leaf_indices[selection_index];
                let shared_span = batch_route.selected_leaf_spans[selection_index];
                let score = batch_route.selected_leaf_scores[selection_index];
                if shared_span.end() >= query_position {
                    continue;
                }
                let expected_span = *self.shared_spans.get(leaf_index).ok_or_else(|| {
                    FractalError::InvalidState(format!(
                        "sealed_token_state_store routed leaf index {leaf_index} is out of bounds for {} sealed spans",
                        self.shared_spans.len()
                    ))
                })?;
                if shared_span != expected_span {
                    return Err(FractalError::InvalidState(format!(
                        "sealed_token_state_store routed span mismatch at leaf {leaf_index}: expected [{}, {}), got [{}, {})",
                        expected_span.start(),
                        expected_span.end(),
                        shared_span.start(),
                        shared_span.end()
                    )));
                }
                by_leaf
                    .entry(leaf_index)
                    .and_modify(|(_, best_score)| {
                        if score > *best_score {
                            *best_score = score;
                        }
                    })
                    .or_insert((shared_span, score));
            }
        }

        let mut ranked = by_leaf
            .into_iter()
            .map(|(leaf_index, (shared_span, score))| (leaf_index, shared_span, score))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.start().cmp(&right.1.start()))
                .then_with(|| left.0.cmp(&right.0))
        });
        Ok(ranked
            .into_iter()
            .take(max_span_count)
            .map(|(leaf_index, _, _)| leaf_index)
            .collect())
    }

    fn select_oracle_leaf_indices(
        &self,
        evidence_span: TokenSpan,
        query_position: usize,
        max_span_count: usize,
    ) -> Vec<usize> {
        self.shared_spans
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, span)| span.end() < query_position && spans_overlap(*span, evidence_span))
            .map(|(leaf_index, _)| leaf_index)
            .take(max_span_count)
            .collect()
    }

    fn select_oracle_exact_tokens(
        &self,
        evidence_span: TokenSpan,
        query_position: usize,
        max_token_count: usize,
    ) -> Result<Vec<(usize, usize, TokenSpan)>, FractalError> {
        let mut selected = Vec::new();
        for (leaf_index, shared_span) in self.shared_spans.iter().copied().enumerate() {
            if shared_span.end() >= query_position || !spans_overlap(shared_span, evidence_span) {
                continue;
            }
            let overlap_start = shared_span.start().max(evidence_span.start());
            let overlap_end = shared_span.end().min(evidence_span.end());
            for absolute_position in overlap_start..overlap_end {
                let token_offset = absolute_position - shared_span.start();
                selected.push((leaf_index, token_offset, shared_span));
                if selected.len() == max_token_count {
                    return Ok(selected);
                }
            }
        }

        Ok(selected)
    }

    fn build_sealed_span_pack_rows(
        &self,
        batch_index: usize,
        selected_leaf_indices: &[usize],
        token_capacity: usize,
        device: &B::Device,
    ) -> Result<GatheredRowBatch<B>, FractalError> {
        let mut row_groups = Vec::new();
        let mut absolute_positions = Vec::with_capacity(token_capacity);
        let mut source_span_starts = Vec::with_capacity(token_capacity);
        let mut source_span_ends = Vec::with_capacity(token_capacity);
        let mut token_mask = Vec::with_capacity(token_capacity);

        for &leaf_index in selected_leaf_indices {
            let shared_span = *self.shared_spans.get(leaf_index).ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "sealed_token_state_store selected leaf index {leaf_index} is out of bounds"
                ))
            })?;
            let rows = self.token_states[leaf_index]
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    0..self.leaf_size,
                    0..self.token_state_dim,
                ])
                .reshape([self.leaf_size, self.token_state_dim]);
            row_groups.push(rows);
            for absolute_position in shared_span.start()..shared_span.end() {
                absolute_positions.push(absolute_position as i64);
                source_span_starts.push(shared_span.start() as i64);
                source_span_ends.push(shared_span.end() as i64);
                token_mask.push(true);
            }
        }

        let active_token_count = selected_leaf_indices.len() * self.leaf_size;
        if active_token_count < token_capacity {
            row_groups.push(Tensor::<B, 2>::zeros(
                [token_capacity - active_token_count, self.token_state_dim],
                device,
            ));
            for _ in active_token_count..token_capacity {
                absolute_positions.push(-1);
                source_span_starts.push(-1);
                source_span_ends.push(-1);
                token_mask.push(false);
            }
        }

        Ok(GatheredRowBatch {
            token_states: Tensor::cat(row_groups, 0).reshape([1, token_capacity, self.token_state_dim]),
            absolute_positions,
            source_span_starts,
            source_span_ends,
            token_mask,
        })
    }

    fn build_exact_token_subset_rows(
        &self,
        batch_index: usize,
        selected_tokens: &[(usize, usize, TokenSpan)],
        token_capacity: usize,
        device: &B::Device,
    ) -> Result<GatheredRowBatch<B>, FractalError> {
        let mut row_groups = Vec::new();
        let mut absolute_positions = Vec::with_capacity(token_capacity);
        let mut source_span_starts = Vec::with_capacity(token_capacity);
        let mut source_span_ends = Vec::with_capacity(token_capacity);
        let mut token_mask = Vec::with_capacity(token_capacity);

        for &(leaf_index, token_offset, shared_span) in selected_tokens {
            let rows = self.token_states[leaf_index]
                .clone()
                .slice([
                    batch_index..batch_index + 1,
                    token_offset..token_offset + 1,
                    0..self.token_state_dim,
                ])
                .reshape([1, self.token_state_dim]);
            row_groups.push(rows);
            absolute_positions.push((shared_span.start() + token_offset) as i64);
            source_span_starts.push(shared_span.start() as i64);
            source_span_ends.push(shared_span.end() as i64);
            token_mask.push(true);
        }

        if selected_tokens.len() < token_capacity {
            row_groups.push(Tensor::<B, 2>::zeros(
                [token_capacity - selected_tokens.len(), self.token_state_dim],
                device,
            ));
            for _ in selected_tokens.len()..token_capacity {
                absolute_positions.push(-1);
                source_span_starts.push(-1);
                source_span_ends.push(-1);
                token_mask.push(false);
            }
        }

        Ok(GatheredRowBatch {
            token_states: Tensor::cat(row_groups, 0).reshape([1, token_capacity, self.token_state_dim]),
            absolute_positions,
            source_span_starts,
            source_span_ends,
            token_mask,
        })
    }

    fn build_inactive_rows(
        &self,
        token_capacity: usize,
        device: &B::Device,
    ) -> GatheredRowBatch<B> {
        GatheredRowBatch {
            token_states: Tensor::<B, 3>::zeros([1, token_capacity, self.token_state_dim], device),
            absolute_positions: vec![-1; token_capacity],
            source_span_starts: vec![-1; token_capacity],
            source_span_ends: vec![-1; token_capacity],
            token_mask: vec![false; token_capacity],
        }
    }
}

fn validate_position_contract<B: Backend>(
    layout: GatheredRetrievalLayout,
    batch_size: usize,
    token_capacity: usize,
    absolute_positions: Tensor<B, 2, Int>,
    source_span_starts: Tensor<B, 2, Int>,
    source_span_ends: Tensor<B, 2, Int>,
    token_mask: Tensor<B, 2, Bool>,
) -> Result<(), FractalError> {
    let absolute_positions =
        tensor_data_to_i64(absolute_positions, "gathered_retrieval.absolute_positions")?;
    let source_span_starts =
        tensor_data_to_i64(source_span_starts, "gathered_retrieval.source_span_starts")?;
    let source_span_ends =
        tensor_data_to_i64(source_span_ends, "gathered_retrieval.source_span_ends")?;
    let token_mask = tensor_data_to_bool(token_mask, "gathered_retrieval.token_mask")?;

    match layout {
        GatheredRetrievalLayout::SealedSpanPacks {
            max_span_count,
            leaf_size,
        } => validate_sealed_span_pack_contract(
            batch_size,
            token_capacity,
            max_span_count,
            leaf_size,
            &absolute_positions,
            &source_span_starts,
            &source_span_ends,
            &token_mask,
        ),
        GatheredRetrievalLayout::ExactTokenSubset { .. } => validate_exact_token_subset_contract(
            batch_size,
            token_capacity,
            &absolute_positions,
            &source_span_starts,
            &source_span_ends,
            &token_mask,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_sealed_span_pack_contract(
    batch_size: usize,
    token_capacity: usize,
    max_span_count: usize,
    leaf_size: usize,
    absolute_positions: &[i64],
    source_span_starts: &[i64],
    source_span_ends: &[i64],
    token_mask: &[bool],
) -> Result<(), FractalError> {
    ensure_match(
        "gathered_retrieval.sealed_span_packs.token_capacity",
        token_capacity,
        max_span_count * leaf_size,
    )?;

    for batch_index in 0..batch_size {
        let mut inactive_span_seen = false;
        let mut seen_spans = BTreeSet::new();
        for span_slot in 0..max_span_count {
            let span_base = batch_index * token_capacity + span_slot * leaf_size;
            let mut active_count = 0usize;
            for token_offset in 0..leaf_size {
                if token_mask[span_base + token_offset] {
                    active_count += 1;
                }
            }

            if active_count == 0 {
                inactive_span_seen = true;
                for token_offset in 0..leaf_size {
                    let flat_index = span_base + token_offset;
                    ensure_inactive_slot("absolute_positions", absolute_positions[flat_index])?;
                    ensure_inactive_slot("source_span_starts", source_span_starts[flat_index])?;
                    ensure_inactive_slot("source_span_ends", source_span_ends[flat_index])?;
                }
                continue;
            }

            if inactive_span_seen {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} must pack active sealed spans before inactive span slots"
                )));
            }
            if active_count != leaf_size {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} span slot {span_slot} must contain either all {leaf_size} tokens or none"
                )));
            }

            let span_start = source_span_starts[span_base];
            let span_end = source_span_ends[span_base];
            if span_start < 0 || span_end < 0 {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} span slot {span_slot} must have non-negative span metadata"
                )));
            }
            if span_end - span_start != leaf_size as i64 {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} span slot {span_slot} must span exactly {leaf_size} tokens, got [{span_start}, {span_end})"
                )));
            }
            if !seen_spans.insert((span_start, span_end)) {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} span slot {span_slot} duplicated sealed span [{span_start}, {span_end})"
                )));
            }

            for token_offset in 0..leaf_size {
                let flat_index = span_base + token_offset;
                let absolute_position = absolute_positions[flat_index];
                if source_span_starts[flat_index] != span_start
                    || source_span_ends[flat_index] != span_end
                {
                    return Err(FractalError::InvalidConfig(format!(
                        "gathered retrieval batch {batch_index} span slot {span_slot} must carry consistent span metadata across all {leaf_size} tokens"
                    )));
                }
                ensure_active_slot_in_span(
                    flat_index,
                    absolute_position,
                    span_start,
                    span_end,
                    Some(span_start + token_offset as i64),
                )?;
            }
        }
    }

    Ok(())
}

fn validate_exact_token_subset_contract(
    batch_size: usize,
    token_capacity: usize,
    absolute_positions: &[i64],
    source_span_starts: &[i64],
    source_span_ends: &[i64],
    token_mask: &[bool],
) -> Result<(), FractalError> {
    for batch_index in 0..batch_size {
        let mut inactive_slot_seen = false;
        let mut seen_positions = BTreeSet::new();
        for slot in 0..token_capacity {
            let flat_index = batch_index * token_capacity + slot;
            if !token_mask[flat_index] {
                inactive_slot_seen = true;
                ensure_inactive_slot("absolute_positions", absolute_positions[flat_index])?;
                ensure_inactive_slot("source_span_starts", source_span_starts[flat_index])?;
                ensure_inactive_slot("source_span_ends", source_span_ends[flat_index])?;
                continue;
            }
            if inactive_slot_seen {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} exact-token subset must pack active tokens before inactive slots"
                )));
            }

            let absolute_position = absolute_positions[flat_index];
            let span_start = source_span_starts[flat_index];
            let span_end = source_span_ends[flat_index];
            ensure_active_slot_in_span(flat_index, absolute_position, span_start, span_end, None)?;
            if !seen_positions.insert(absolute_position) {
                return Err(FractalError::InvalidConfig(format!(
                    "gathered retrieval batch {batch_index} exact-token subset duplicated absolute position {absolute_position}"
                )));
            }
        }
    }

    Ok(())
}

fn ensure_active_slot_in_span(
    flat_index: usize,
    absolute_position: i64,
    span_start: i64,
    span_end: i64,
    expected_absolute_position: Option<i64>,
) -> Result<(), FractalError> {
    if span_start < 0 || span_end < 0 || absolute_position < 0 {
        return Err(FractalError::InvalidConfig(format!(
            "gathered retrieval active slot {flat_index} must have non-negative positions"
        )));
    }
    if span_end <= span_start {
        return Err(FractalError::InvalidConfig(format!(
            "gathered retrieval active slot {flat_index} must have span_end > span_start"
        )));
    }
    if absolute_position < span_start || absolute_position >= span_end {
        return Err(FractalError::InvalidConfig(format!(
            "gathered retrieval active slot {flat_index} absolute position {absolute_position} must lie within source span [{span_start}, {span_end})"
        )));
    }
    if let Some(expected_absolute_position) = expected_absolute_position {
        if absolute_position != expected_absolute_position {
            return Err(FractalError::InvalidConfig(format!(
                "gathered retrieval active slot {flat_index} must be packed in sealed-span order; expected absolute position {expected_absolute_position}, got {absolute_position}"
            )));
        }
    }

    Ok(())
}

fn ensure_inactive_slot(name: &str, value: i64) -> Result<(), FractalError> {
    if value != -1 {
        return Err(FractalError::InvalidConfig(format!(
            "gathered retrieval inactive slot {name} must be -1, got {value}"
        )));
    }

    Ok(())
}

fn spans_overlap(left: TokenSpan, right: TokenSpan) -> bool {
    left.start() < right.end() && right.start() < left.end()
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

    #[test]
    fn gathered_retrieval_context_tracks_candidate_recall() {
        let device = <TestBackend as Backend>::Device::default();
        let context = GatheredRetrievalContext::from_tensors(
            GatheredRetrievalProvenance::OracleExactTokenSubset,
            GatheredRetrievalLayout::ExactTokenSubset { max_token_count: 4 },
            Tensor::<TestBackend, 3>::zeros([1, 4, 3], &device),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0i64, 1, 16, -1], [1, 4]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0i64, 0, 16, -1], [1, 4]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![2i64, 2, 32, -1], [1, 4]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(vec![true, true, true, false], [1, 4]),
                &device,
            ),
        )
        .unwrap();

        let recall = context
            .candidate_recall_for_span(TokenSpan::new(0, 2).unwrap())
            .unwrap();

        assert_eq!(recall.len(), 1);
        assert!(recall[0].evidence_span_recalled);
        assert_eq!(recall[0].gathered_evidence_token_count, 2);
        assert_eq!(recall[0].evidence_token_count, 2);
        assert_eq!(recall[0].evidence_token_recall(), 1.0);
    }

    #[test]
    fn gathered_retrieval_context_rejects_positions_outside_source_span() {
        let device = <TestBackend as Backend>::Device::default();
        let error = GatheredRetrievalContext::from_tensors(
            GatheredRetrievalProvenance::OracleExactTokenSubset,
            GatheredRetrievalLayout::ExactTokenSubset { max_token_count: 1 },
            Tensor::<TestBackend, 3>::zeros([1, 1, 3], &device),
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![17i64], [1, 1]), &device),
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![0i64], [1, 1]), &device),
            Tensor::<TestBackend, 2, Int>::from_data(TensorData::new(vec![16i64], [1, 1]), &device),
            Tensor::<TestBackend, 2, Bool>::from_data(TensorData::new(vec![true], [1, 1]), &device),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("absolute position 17"))
        );
    }

    #[test]
    fn gathered_retrieval_context_rejects_partial_sealed_span_pack() {
        let device = <TestBackend as Backend>::Device::default();
        let error = GatheredRetrievalContext::from_tensors(
            GatheredRetrievalProvenance::Routed,
            GatheredRetrievalLayout::SealedSpanPacks {
                max_span_count: 1,
                leaf_size: 2,
            },
            Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0i64, -1], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0i64, -1], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![2i64, -1], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(vec![true, false], [1, 2]),
                &device,
            ),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("either all 2 tokens or none"))
        );
    }

    #[test]
    fn gathered_retrieval_context_rejects_duplicate_exact_subset_positions() {
        let device = <TestBackend as Backend>::Device::default();
        let error = GatheredRetrievalContext::from_tensors(
            GatheredRetrievalProvenance::OracleExactTokenSubset,
            GatheredRetrievalLayout::ExactTokenSubset { max_token_count: 2 },
            Tensor::<TestBackend, 3>::zeros([1, 2, 3], &device),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![7i64, 7], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![0i64, 0], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Int>::from_data(
                TensorData::new(vec![8i64, 8], [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2, Bool>::from_data(
                TensorData::new(vec![true, true], [1, 2]),
                &device,
            ),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("duplicated absolute position 7"))
        );
    }
}
