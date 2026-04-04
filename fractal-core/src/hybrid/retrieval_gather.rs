use std::collections::BTreeSet;

use burn::tensor::{backend::Backend, Bool, Int, Tensor};

use crate::{error::FractalError, v2::TokenSpan};

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
