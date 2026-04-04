use burn::prelude::ElementConversion;
use burn::record::{PrecisionSettings, Record};
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::FractalError;

use super::{
    leaf::LeafSummarizer,
    local_trunk::LocalTrunkShape,
    model::FractalV2ModelShape,
    router::FractalRouterHeadShape,
    tree::{TreeMergeCell, TreeNodeBatch, TreeSummaryDiagnostics},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchTimelineMode {
    LockstepSharedTimeline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSpan {
    start: usize,
    end: usize,
}

impl TokenSpan {
    pub fn new(start: usize, end: usize) -> Result<Self, FractalError> {
        if end < start {
            return Err(FractalError::InvalidConfig(format!(
                "token span end {end} must be greater than or equal to start {start}"
            )));
        }

        Ok(Self { start, end })
    }

    pub fn empty_at(start: usize) -> Self {
        Self { start, end: start }
    }

    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn end(&self) -> usize {
        self.end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalPolicy {
    beam_width: usize,
    top_k_reads: usize,
    allow_early_stop: bool,
}

impl RetrievalPolicy {
    pub fn from_router_shape(shape: FractalRouterHeadShape) -> Result<Self, FractalError> {
        let policy = Self {
            beam_width: shape.beam_width,
            top_k_reads: shape.top_leaf_reads,
            allow_early_stop: shape.allow_early_stop,
        };
        policy.validate()?;
        Ok(policy)
    }

    pub fn beam_width(&self) -> usize {
        self.beam_width
    }

    pub fn top_k_reads(&self) -> usize {
        self.top_k_reads
    }

    pub fn allow_early_stop(&self) -> bool {
        self.allow_early_stop
    }

    fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("retrieval_policy.beam_width", self.beam_width)?;
        ensure_nonzero("retrieval_policy.top_k_reads", self.top_k_reads)?;
        if self.allow_early_stop {
            return Err(FractalError::InvalidConfig(
                "retrieval_policy.allow_early_stop must remain false in v1".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeCheckpointPolicy {
    FixedLeafSize { tokens_per_leaf: usize },
}

impl MergeCheckpointPolicy {
    pub fn tokens_per_leaf(&self) -> usize {
        match self {
            Self::FixedLeafSize { tokens_per_leaf } => *tokens_per_leaf,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero(
            "merge_checkpoint_policy.tokens_per_leaf",
            self.tokens_per_leaf(),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FractalV2StateLayout {
    batch_size: usize,
    batch_timeline_mode: BatchTimelineMode,
    root_count: usize,
    root_state_dim: usize,
    root_readout_dim: usize,
    leaf_size: usize,
    summary_dim: usize,
    key_dim: usize,
    value_dim: usize,
    token_cache_key_dim: usize,
    token_cache_value_dim: usize,
    scale_embedding_dim: usize,
}

impl FractalV2StateLayout {
    pub fn from_model_shape(
        shape: FractalV2ModelShape,
        batch_size: usize,
    ) -> Result<Self, FractalError> {
        shape.validate()?;
        let layout = Self {
            batch_size,
            batch_timeline_mode: BatchTimelineMode::LockstepSharedTimeline,
            root_count: shape.local_trunk.root_count,
            root_state_dim: shape.local_trunk.root_state_dim,
            root_readout_dim: shape.local_trunk.root_readout_dim,
            leaf_size: shape.local_trunk.leaf_size,
            summary_dim: shape.leaf_summarizer.summary_dim,
            key_dim: shape.leaf_summarizer.key_dim,
            value_dim: shape.leaf_summarizer.value_dim,
            token_cache_key_dim: shape.leaf_summarizer.token_cache_key_dim,
            token_cache_value_dim: shape.leaf_summarizer.token_cache_value_dim,
            scale_embedding_dim: shape.tree_merge_cell.scale_embedding_dim,
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        ensure_nonzero("state_layout.batch_size", self.batch_size)?;
        ensure_nonzero("state_layout.root_count", self.root_count)?;
        ensure_nonzero("state_layout.root_state_dim", self.root_state_dim)?;
        ensure_nonzero("state_layout.root_readout_dim", self.root_readout_dim)?;
        ensure_nonzero("state_layout.leaf_size", self.leaf_size)?;
        ensure_nonzero("state_layout.summary_dim", self.summary_dim)?;
        ensure_nonzero("state_layout.key_dim", self.key_dim)?;
        ensure_nonzero("state_layout.value_dim", self.value_dim)?;
        ensure_nonzero("state_layout.token_cache_key_dim", self.token_cache_key_dim)?;
        ensure_nonzero(
            "state_layout.token_cache_value_dim",
            self.token_cache_value_dim,
        )?;
        ensure_nonzero("state_layout.scale_embedding_dim", self.scale_embedding_dim)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn batch_timeline_mode(&self) -> BatchTimelineMode {
        self.batch_timeline_mode
    }

    pub fn root_count(&self) -> usize {
        self.root_count
    }

    pub fn root_state_dim(&self) -> usize {
        self.root_state_dim
    }

    pub fn root_readout_dim(&self) -> usize {
        self.root_readout_dim
    }

    pub fn leaf_size(&self) -> usize {
        self.leaf_size
    }

    pub fn summary_dim(&self) -> usize {
        self.summary_dim
    }

    pub fn key_dim(&self) -> usize {
        self.key_dim
    }

    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    pub fn token_cache_key_dim(&self) -> usize {
        self.token_cache_key_dim
    }

    pub fn token_cache_value_dim(&self) -> usize {
        self.token_cache_value_dim
    }

    pub fn scale_embedding_dim(&self) -> usize {
        self.scale_embedding_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiRootStateShape {
    pub batch_size: usize,
    pub root_count: usize,
    pub recurrent_dim: usize,
    pub intent_dim: usize,
}

#[derive(Debug, Clone)]
pub struct MultiRootStateRecord<B: Backend> {
    pub recurrent: Tensor<B, 3>,
    pub read_intent: Tensor<B, 3>,
    pub write_intent: Tensor<B, 3>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRootStateRecordItem<Tensor3Item> {
    pub recurrent: Tensor3Item,
    pub read_intent: Tensor3Item,
    pub write_intent: Tensor3Item,
}

#[derive(Debug, Clone)]
pub struct MultiRootState<B: Backend> {
    recurrent: Tensor<B, 3>,
    read_intent: Tensor<B, 3>,
    write_intent: Tensor<B, 3>,
}

impl<B: Backend> MultiRootState<B> {
    pub fn from_tensors(
        recurrent: Tensor<B, 3>,
        read_intent: Tensor<B, 3>,
        write_intent: Tensor<B, 3>,
    ) -> Result<Self, FractalError> {
        let [batch_size, root_count, recurrent_dim] = recurrent.dims();
        let [read_batch, read_root_count, intent_dim] = read_intent.dims();
        let [write_batch, write_root_count, write_intent_dim] = write_intent.dims();

        ensure_nonzero("multi_root.batch_size", batch_size)?;
        ensure_nonzero("multi_root.root_count", root_count)?;
        ensure_match("multi_root.read_intent.batch_size", read_batch, batch_size)?;
        ensure_match(
            "multi_root.read_intent.root_count",
            read_root_count,
            root_count,
        )?;
        ensure_match(
            "multi_root.write_intent.batch_size",
            write_batch,
            batch_size,
        )?;
        ensure_match(
            "multi_root.write_intent.root_count",
            write_root_count,
            root_count,
        )?;
        ensure_match(
            "multi_root.write_intent.intent_dim",
            write_intent_dim,
            intent_dim,
        )?;
        ensure_nonzero("multi_root.recurrent_dim", recurrent_dim)?;
        ensure_nonzero("multi_root.intent_dim", intent_dim)?;

        Ok(Self {
            recurrent,
            read_intent,
            write_intent,
        })
    }

    pub fn zeros_for_local_trunk(
        batch_size: usize,
        shape: LocalTrunkShape,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        ensure_nonzero("multi_root.batch_size", batch_size)?;
        ensure_nonzero("multi_root.root_count", shape.root_count)?;
        ensure_nonzero("multi_root.recurrent_dim", shape.root_state_dim)?;
        ensure_nonzero("multi_root.intent_dim", shape.root_readout_dim)?;

        Ok(Self {
            recurrent: Tensor::<B, 3>::zeros(
                [batch_size, shape.root_count, shape.root_state_dim],
                device,
            ),
            read_intent: Tensor::<B, 3>::zeros(
                [batch_size, shape.root_count, shape.root_readout_dim],
                device,
            ),
            write_intent: Tensor::<B, 3>::zeros(
                [batch_size, shape.root_count, shape.root_readout_dim],
                device,
            ),
        })
    }

    pub fn zeros(layout: FractalV2StateLayout, device: &B::Device) -> Self {
        Self {
            recurrent: Tensor::<B, 3>::zeros(
                [layout.batch_size, layout.root_count, layout.root_state_dim],
                device,
            ),
            read_intent: Tensor::<B, 3>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.root_readout_dim,
                ],
                device,
            ),
            write_intent: Tensor::<B, 3>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.root_readout_dim,
                ],
                device,
            ),
        }
    }

    pub fn from_record(
        record: MultiRootStateRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        ensure_dims3(
            "multi_root.recurrent",
            record.recurrent.dims(),
            [layout.batch_size, layout.root_count, layout.root_state_dim],
        )?;
        ensure_dims3(
            "multi_root.read_intent",
            record.read_intent.dims(),
            [
                layout.batch_size,
                layout.root_count,
                layout.root_readout_dim,
            ],
        )?;
        ensure_dims3(
            "multi_root.write_intent",
            record.write_intent.dims(),
            [
                layout.batch_size,
                layout.root_count,
                layout.root_readout_dim,
            ],
        )?;

        Ok(Self {
            recurrent: record.recurrent,
            read_intent: record.read_intent,
            write_intent: record.write_intent,
        })
    }

    pub fn into_record(&self) -> MultiRootStateRecord<B> {
        MultiRootStateRecord {
            recurrent: self.recurrent.clone(),
            read_intent: self.read_intent.clone(),
            write_intent: self.write_intent.clone(),
        }
    }

    pub fn shape(&self) -> MultiRootStateShape {
        let [batch_size, root_count, recurrent_dim] = self.recurrent.dims();
        let [_, _, intent_dim] = self.read_intent.dims();
        MultiRootStateShape {
            batch_size,
            root_count,
            recurrent_dim,
            intent_dim,
        }
    }

    pub fn recurrent(&self) -> Tensor<B, 3> {
        self.recurrent.clone()
    }

    pub fn read_intent(&self) -> Tensor<B, 3> {
        self.read_intent.clone()
    }

    pub fn write_intent(&self) -> Tensor<B, 3> {
        self.write_intent.clone()
    }
}

impl<B: Backend> Record<B> for MultiRootStateRecord<B> {
    type Item<S: PrecisionSettings> =
        MultiRootStateRecordItem<<Tensor<B, 3> as Record<B>>::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        MultiRootStateRecordItem {
            recurrent: Record::<B>::into_item::<S>(self.recurrent),
            read_intent: Record::<B>::into_item::<S>(self.read_intent),
            write_intent: Record::<B>::into_item::<S>(self.write_intent),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            recurrent: Record::<B>::from_item::<S>(item.recurrent, device),
            read_intent: Record::<B>::from_item::<S>(item.read_intent, device),
            write_intent: Record::<B>::from_item::<S>(item.write_intent, device),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveLeafStateShape {
    pub batch_size: usize,
    pub root_count: usize,
    pub tokens_per_leaf: usize,
    pub readout_dim: usize,
    pub batch_timeline_mode: BatchTimelineMode,
    pub shared_span: TokenSpan,
    pub shared_valid_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct LiveLeafStateRecord<B: Backend> {
    pub token_readouts: Tensor<B, 4>,
    pub shared_span: TokenSpan,
    pub shared_valid_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveLeafStateRecordItem<Tensor4Item> {
    pub token_readouts: Tensor4Item,
    pub shared_span: TokenSpan,
    pub shared_valid_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct LiveLeafState<B: Backend> {
    token_readouts: Tensor<B, 4>,
    shared_span: TokenSpan,
    shared_valid_tokens: usize,
}

impl<B: Backend> LiveLeafState<B> {
    pub fn empty(layout: FractalV2StateLayout, device: &B::Device) -> Self {
        Self {
            token_readouts: Tensor::<B, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                device,
            ),
            shared_span: TokenSpan::empty_at(0),
            shared_valid_tokens: 0,
        }
    }

    pub fn from_record(
        record: LiveLeafStateRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        ensure_dims4(
            "live_leaf.token_readouts",
            record.token_readouts.dims(),
            [
                layout.batch_size,
                layout.root_count,
                layout.leaf_size,
                layout.root_readout_dim,
            ],
        )?;
        validate_token_span("live_leaf.shared_span", record.shared_span)?;
        ensure_at_most(
            "live_leaf.shared_valid_tokens",
            record.shared_valid_tokens,
            layout.leaf_size,
        )?;
        ensure_match(
            "live_leaf.shared_valid_tokens",
            record.shared_valid_tokens,
            record.shared_span.len(),
        )?;
        ensure_multiple_of(
            "live_leaf.shared_span.start",
            record.shared_span.start(),
            layout.leaf_size,
        )?;
        validate_live_leaf_zero_tail(&record.token_readouts, record.shared_valid_tokens)?;

        Ok(Self {
            token_readouts: record.token_readouts,
            shared_span: record.shared_span,
            shared_valid_tokens: record.shared_valid_tokens,
        })
    }

    pub fn into_record(&self) -> LiveLeafStateRecord<B> {
        LiveLeafStateRecord {
            token_readouts: self.token_readouts.clone(),
            shared_span: self.shared_span,
            shared_valid_tokens: self.shared_valid_tokens,
        }
    }

    pub fn shape(&self, batch_timeline_mode: BatchTimelineMode) -> LiveLeafStateShape {
        let [batch_size, root_count, tokens_per_leaf, readout_dim] = self.token_readouts.dims();
        LiveLeafStateShape {
            batch_size,
            root_count,
            tokens_per_leaf,
            readout_dim,
            batch_timeline_mode,
            shared_span: self.shared_span,
            shared_valid_tokens: self.shared_valid_tokens,
        }
    }

    pub fn token_readouts(&self) -> Tensor<B, 4> {
        self.token_readouts.clone()
    }

    pub fn shared_span(&self) -> TokenSpan {
        self.shared_span
    }

    pub fn shared_valid_tokens(&self) -> usize {
        self.shared_valid_tokens
    }

    pub fn append_root_readouts(
        &mut self,
        root_readouts: Tensor<B, 3>,
    ) -> Result<Option<(Tensor<B, 4>, TokenSpan)>, FractalError> {
        let [batch_size, root_count, readout_dim] = root_readouts.dims();
        let [expected_batch_size, expected_root_count, leaf_size, expected_readout_dim] =
            self.token_readouts.dims();
        ensure_match(
            "live_leaf.append.batch_size",
            batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "live_leaf.append.root_count",
            root_count,
            expected_root_count,
        )?;
        ensure_match(
            "live_leaf.append.readout_dim",
            readout_dim,
            expected_readout_dim,
        )?;
        ensure_at_most(
            "live_leaf.append.shared_valid_tokens",
            self.shared_valid_tokens,
            leaf_size,
        )?;
        if self.shared_valid_tokens == leaf_size {
            return Err(FractalError::InvalidState(
                "live leaf is full before append; it should have been sealed already".to_string(),
            ));
        }

        let token_index = self.shared_valid_tokens;
        self.token_readouts = self.token_readouts.clone().slice_assign(
            [
                0..batch_size,
                0..root_count,
                token_index..token_index + 1,
                0..readout_dim,
            ],
            root_readouts.reshape([batch_size, root_count, 1, readout_dim]),
        );
        self.shared_valid_tokens += 1;
        self.shared_span = TokenSpan::new(self.shared_span.start(), self.shared_span.end() + 1)?;

        if self.shared_valid_tokens < leaf_size {
            return Ok(None);
        }

        let sealed_token_readouts = self.token_readouts.clone();
        let sealed_span = self.shared_span;
        self.token_readouts = Tensor::<B, 4>::zeros(
            [batch_size, root_count, leaf_size, readout_dim],
            &sealed_token_readouts.device(),
        );
        self.shared_valid_tokens = 0;
        self.shared_span = TokenSpan::empty_at(sealed_span.end());

        Ok(Some((sealed_token_readouts, sealed_span)))
    }
}

impl<B: Backend> Record<B> for LiveLeafStateRecord<B> {
    type Item<S: PrecisionSettings> = LiveLeafStateRecordItem<<Tensor<B, 4> as Record<B>>::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        LiveLeafStateRecordItem {
            token_readouts: Record::<B>::into_item::<S>(self.token_readouts),
            shared_span: self.shared_span,
            shared_valid_tokens: self.shared_valid_tokens,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            token_readouts: Record::<B>::from_item::<S>(item.token_readouts, device),
            shared_span: item.shared_span,
            shared_valid_tokens: item.shared_valid_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SealedLeafMaterialization<B: Backend> {
    leaf_index: usize,
    shared_span: TokenSpan,
    summary: Tensor<B, 2>,
    key: Tensor<B, 2>,
    value: Tensor<B, 2>,
    token_keys: Tensor<B, 3>,
    token_values: Tensor<B, 3>,
    token_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> SealedLeafMaterialization<B> {
    pub fn leaf_index(&self) -> usize {
        self.leaf_index
    }

    pub fn shared_span(&self) -> TokenSpan {
        self.shared_span
    }

    pub fn summary(&self) -> Tensor<B, 2> {
        self.summary.clone()
    }

    pub fn key(&self) -> Tensor<B, 2> {
        self.key.clone()
    }

    pub fn value(&self) -> Tensor<B, 2> {
        self.value.clone()
    }

    pub fn token_keys(&self) -> Tensor<B, 3> {
        self.token_keys.clone()
    }

    pub fn token_values(&self) -> Tensor<B, 3> {
        self.token_values.clone()
    }

    pub fn token_mask(&self) -> Tensor<B, 2, Bool> {
        self.token_mask.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafSummaryStoreShape {
    pub batch_size: usize,
    pub leaf_count: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub batch_timeline_mode: BatchTimelineMode,
}

#[derive(Debug, Clone)]
pub struct LeafSummaryStoreRecord<B: Backend> {
    pub summaries: Tensor<B, 3>,
    pub keys: Tensor<B, 3>,
    pub values: Tensor<B, 3>,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafSummaryStoreRecordItem<Tensor3Item> {
    pub summaries: Tensor3Item,
    pub keys: Tensor3Item,
    pub values: Tensor3Item,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone)]
pub struct LeafSummaryStore<B: Backend> {
    summaries: Tensor<B, 3>,
    keys: Tensor<B, 3>,
    values: Tensor<B, 3>,
    shared_spans: Vec<TokenSpan>,
}

impl<B: Backend> LeafSummaryStore<B> {
    pub fn empty(layout: FractalV2StateLayout, device: &B::Device) -> Self {
        Self {
            summaries: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.summary_dim], device),
            keys: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.key_dim], device),
            values: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.value_dim], device),
            shared_spans: Vec::new(),
        }
    }

    pub fn from_record(
        record: LeafSummaryStoreRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        let summary_dims = record.summaries.dims();
        let leaf_count = summary_dims[1];
        ensure_dims3(
            "leaf_summary.summaries",
            summary_dims,
            [layout.batch_size, leaf_count, layout.summary_dim],
        )?;
        ensure_dims3(
            "leaf_summary.keys",
            record.keys.dims(),
            [layout.batch_size, leaf_count, layout.key_dim],
        )?;
        ensure_dims3(
            "leaf_summary.values",
            record.values.dims(),
            [layout.batch_size, leaf_count, layout.value_dim],
        )?;
        validate_fixed_width_prefix_spans(
            "leaf_summary.shared_spans",
            &record.shared_spans,
            layout.leaf_size,
        )?;
        ensure_match(
            "leaf_summary.shared_span_count",
            record.shared_spans.len(),
            leaf_count,
        )?;

        Ok(Self {
            summaries: record.summaries,
            keys: record.keys,
            values: record.values,
            shared_spans: record.shared_spans,
        })
    }

    pub fn into_record(&self) -> LeafSummaryStoreRecord<B> {
        LeafSummaryStoreRecord {
            summaries: self.summaries.clone(),
            keys: self.keys.clone(),
            values: self.values.clone(),
            shared_spans: self.shared_spans.clone(),
        }
    }

    pub fn shape(&self, batch_timeline_mode: BatchTimelineMode) -> LeafSummaryStoreShape {
        let [batch_size, leaf_count, summary_dim] = self.summaries.dims();
        let [_, _, key_dim] = self.keys.dims();
        let [_, _, value_dim] = self.values.dims();
        LeafSummaryStoreShape {
            batch_size,
            leaf_count,
            summary_dim,
            key_dim,
            value_dim,
            batch_timeline_mode,
        }
    }

    pub fn summaries(&self) -> Tensor<B, 3> {
        self.summaries.clone()
    }

    pub fn keys(&self) -> Tensor<B, 3> {
        self.keys.clone()
    }

    pub fn values(&self) -> Tensor<B, 3> {
        self.values.clone()
    }

    pub fn shared_spans(&self) -> &[TokenSpan] {
        &self.shared_spans
    }

    pub(crate) fn push_sealed_leaf(
        &mut self,
        summary: Tensor<B, 2>,
        key: Tensor<B, 2>,
        value: Tensor<B, 2>,
        tokens_per_leaf: usize,
        shared_span: TokenSpan,
    ) -> Result<usize, FractalError> {
        let [batch_size, summary_dim] = summary.dims();
        let [key_batch_size, key_dim] = key.dims();
        let [value_batch_size, value_dim] = value.dims();
        let [expected_batch_size, leaf_count, expected_summary_dim] = self.summaries.dims();
        let [_, _, expected_key_dim] = self.keys.dims();
        let [_, _, expected_value_dim] = self.values.dims();
        ensure_match(
            "leaf_summary.push.batch_size",
            batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_summary.push.key_batch_size",
            key_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_summary.push.value_batch_size",
            value_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_summary.push.summary_dim",
            summary_dim,
            expected_summary_dim,
        )?;
        ensure_match("leaf_summary.push.key_dim", key_dim, expected_key_dim)?;
        ensure_match("leaf_summary.push.value_dim", value_dim, expected_value_dim)?;
        ensure_nonzero("leaf_summary.push.tokens_per_leaf", tokens_per_leaf)?;
        ensure_match(
            "leaf_summary.push.shared_span.len",
            shared_span.len(),
            tokens_per_leaf,
        )?;

        let leaf_index = leaf_count;
        let expected_span = TokenSpan::new(
            leaf_index.checked_mul(tokens_per_leaf).ok_or_else(|| {
                FractalError::InvalidState(
                    "sealed leaf index overflow while computing expected span".to_string(),
                )
            })?,
            (leaf_index + 1)
                .checked_mul(tokens_per_leaf)
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "sealed leaf index overflow while computing expected span".to_string(),
                    )
                })?,
        )?;
        if shared_span != expected_span {
            return Err(FractalError::InvalidState(format!(
                "sealed leaf span mismatch: expected [{}, {}), got [{}, {})",
                expected_span.start(),
                expected_span.end(),
                shared_span.start(),
                shared_span.end()
            )));
        }

        self.summaries = Tensor::cat(
            vec![
                self.summaries.clone(),
                summary.reshape([batch_size, 1, summary_dim]),
            ],
            1,
        );
        self.keys = Tensor::cat(
            vec![self.keys.clone(), key.reshape([batch_size, 1, key_dim])],
            1,
        );
        self.values = Tensor::cat(
            vec![
                self.values.clone(),
                value.reshape([batch_size, 1, value_dim]),
            ],
            1,
        );
        self.shared_spans.push(shared_span);

        Ok(leaf_index)
    }
}

impl<B: Backend> Record<B> for LeafSummaryStoreRecord<B> {
    type Item<S: PrecisionSettings> =
        LeafSummaryStoreRecordItem<<Tensor<B, 3> as Record<B>>::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        LeafSummaryStoreRecordItem {
            summaries: Record::<B>::into_item::<S>(self.summaries),
            keys: Record::<B>::into_item::<S>(self.keys),
            values: Record::<B>::into_item::<S>(self.values),
            shared_spans: self.shared_spans,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            summaries: Record::<B>::from_item::<S>(item.summaries, device),
            keys: Record::<B>::from_item::<S>(item.keys, device),
            values: Record::<B>::from_item::<S>(item.values, device),
            shared_spans: item.shared_spans,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeLevelStoreShape {
    pub batch_size: usize,
    pub node_count: usize,
    pub summary_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub level: usize,
    pub batch_timeline_mode: BatchTimelineMode,
}

#[derive(Debug, Clone)]
pub struct TreeLevelStoreRecord<B: Backend> {
    pub summaries: Tensor<B, 3>,
    pub keys: Tensor<B, 3>,
    pub values: Tensor<B, 3>,
    pub level: usize,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeLevelStoreRecordItem<Tensor3Item> {
    pub summaries: Tensor3Item,
    pub keys: Tensor3Item,
    pub values: Tensor3Item,
    pub level: usize,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone)]
pub struct TreeLevelStore<B: Backend> {
    summaries: Tensor<B, 3>,
    keys: Tensor<B, 3>,
    values: Tensor<B, 3>,
    level: usize,
    shared_spans: Vec<TokenSpan>,
}

impl<B: Backend> TreeLevelStore<B> {
    pub fn empty(level: usize, layout: FractalV2StateLayout, device: &B::Device) -> Self {
        Self {
            summaries: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.summary_dim], device),
            keys: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.key_dim], device),
            values: Tensor::<B, 3>::zeros([layout.batch_size, 0, layout.value_dim], device),
            level,
            shared_spans: Vec::new(),
        }
    }

    pub fn from_record(
        record: TreeLevelStoreRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        let summary_dims = record.summaries.dims();
        let node_count = summary_dims[1];
        ensure_dims3(
            "tree_level.summaries",
            summary_dims,
            [layout.batch_size, node_count, layout.summary_dim],
        )?;
        ensure_dims3(
            "tree_level.keys",
            record.keys.dims(),
            [layout.batch_size, node_count, layout.key_dim],
        )?;
        ensure_dims3(
            "tree_level.values",
            record.values.dims(),
            [layout.batch_size, node_count, layout.value_dim],
        )?;
        validate_contiguous_prefix_spans(
            &format!("tree_level[{}].shared_spans", record.level),
            &record.shared_spans,
        )?;
        ensure_match(
            "tree_level.shared_span_count",
            record.shared_spans.len(),
            node_count,
        )?;

        Ok(Self {
            summaries: record.summaries,
            keys: record.keys,
            values: record.values,
            level: record.level,
            shared_spans: record.shared_spans,
        })
    }

    pub fn into_record(&self) -> TreeLevelStoreRecord<B> {
        TreeLevelStoreRecord {
            summaries: self.summaries.clone(),
            keys: self.keys.clone(),
            values: self.values.clone(),
            level: self.level,
            shared_spans: self.shared_spans.clone(),
        }
    }

    pub fn shape(&self, batch_timeline_mode: BatchTimelineMode) -> TreeLevelStoreShape {
        let [batch_size, node_count, summary_dim] = self.summaries.dims();
        let [_, _, key_dim] = self.keys.dims();
        let [_, _, value_dim] = self.values.dims();
        TreeLevelStoreShape {
            batch_size,
            node_count,
            summary_dim,
            key_dim,
            value_dim,
            level: self.level,
            batch_timeline_mode,
        }
    }

    pub fn summaries(&self) -> Tensor<B, 3> {
        self.summaries.clone()
    }

    pub fn keys(&self) -> Tensor<B, 3> {
        self.keys.clone()
    }

    pub fn values(&self) -> Tensor<B, 3> {
        self.values.clone()
    }

    pub fn level(&self) -> usize {
        self.level
    }

    pub fn shared_spans(&self) -> &[TokenSpan] {
        &self.shared_spans
    }

    pub fn node_count(&self) -> usize {
        self.shared_spans.len()
    }

    pub fn from_parts(
        layout: FractalV2StateLayout,
        level: usize,
        summaries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
        shared_spans: Vec<TokenSpan>,
    ) -> Result<Self, FractalError> {
        Self::from_record(
            TreeLevelStoreRecord {
                summaries,
                keys,
                values,
                level,
                shared_spans,
            },
            layout,
        )
    }

    fn node(&self, index: usize) -> Result<TreeNodeBatch<B>, FractalError> {
        if index >= self.shared_spans.len() {
            return Err(FractalError::InvalidState(format!(
                "tree level {} node index {index} is out of bounds for {} nodes",
                self.level,
                self.shared_spans.len()
            )));
        }
        let [batch_size, _, summary_dim] = self.summaries.dims();
        let [_, _, key_dim] = self.keys.dims();
        let [_, _, value_dim] = self.values.dims();

        TreeNodeBatch::from_tensors(
            self.summaries
                .clone()
                .narrow(1, index, 1)
                .reshape([batch_size, summary_dim]),
            self.keys
                .clone()
                .narrow(1, index, 1)
                .reshape([batch_size, key_dim]),
            self.values
                .clone()
                .narrow(1, index, 1)
                .reshape([batch_size, value_dim]),
        )
    }

    fn push_node(
        &mut self,
        node: TreeNodeBatch<B>,
        shared_span: TokenSpan,
    ) -> Result<usize, FractalError> {
        let [batch_size, summary_dim] = node.summary().dims();
        let [key_batch_size, key_dim] = node.key().dims();
        let [value_batch_size, value_dim] = node.value().dims();
        let [expected_batch_size, node_count, expected_summary_dim] = self.summaries.dims();
        let [_, _, expected_key_dim] = self.keys.dims();
        let [_, _, expected_value_dim] = self.values.dims();
        ensure_match(
            "tree_level.push.batch_size",
            batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.push.key_batch_size",
            key_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.push.value_batch_size",
            value_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.push.summary_dim",
            summary_dim,
            expected_summary_dim,
        )?;
        ensure_match("tree_level.push.key_dim", key_dim, expected_key_dim)?;
        ensure_match("tree_level.push.value_dim", value_dim, expected_value_dim)?;
        validate_token_span("tree_level.push.shared_span", shared_span)?;

        let expected_start = self
            .shared_spans
            .last()
            .map(TokenSpan::end)
            .unwrap_or_default();
        ensure_match(
            "tree_level.push.shared_span.start",
            shared_span.start(),
            expected_start,
        )?;

        self.summaries = Tensor::cat(
            vec![
                self.summaries.clone(),
                node.summary().reshape([batch_size, 1, summary_dim]),
            ],
            1,
        );
        self.keys = Tensor::cat(
            vec![
                self.keys.clone(),
                node.key().reshape([batch_size, 1, key_dim]),
            ],
            1,
        );
        self.values = Tensor::cat(
            vec![
                self.values.clone(),
                node.value().reshape([batch_size, 1, value_dim]),
            ],
            1,
        );
        self.shared_spans.push(shared_span);

        Ok(node_count)
    }

    fn replace_last_node(
        &mut self,
        node: TreeNodeBatch<B>,
        shared_span: TokenSpan,
    ) -> Result<usize, FractalError> {
        let [batch_size, summary_dim] = node.summary().dims();
        let [key_batch_size, key_dim] = node.key().dims();
        let [value_batch_size, value_dim] = node.value().dims();
        let [expected_batch_size, node_count, expected_summary_dim] = self.summaries.dims();
        let [_, _, expected_key_dim] = self.keys.dims();
        let [_, _, expected_value_dim] = self.values.dims();
        ensure_match(
            "tree_level.replace_last.batch_size",
            batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.replace_last.key_batch_size",
            key_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.replace_last.value_batch_size",
            value_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "tree_level.replace_last.summary_dim",
            summary_dim,
            expected_summary_dim,
        )?;
        ensure_match("tree_level.replace_last.key_dim", key_dim, expected_key_dim)?;
        ensure_match(
            "tree_level.replace_last.value_dim",
            value_dim,
            expected_value_dim,
        )?;
        validate_token_span("tree_level.replace_last.shared_span", shared_span)?;
        if node_count == 0 {
            return Err(FractalError::InvalidState(format!(
                "tree level {} cannot replace the last node of an empty level",
                self.level
            )));
        }

        let last_index = node_count - 1;
        let previous_span = self.shared_spans[last_index];
        ensure_match(
            "tree_level.replace_last.shared_span.start",
            shared_span.start(),
            previous_span.start(),
        )?;
        if shared_span.end() < previous_span.end() {
            return Err(FractalError::InvalidState(format!(
                "tree level {} right frontier span cannot shrink from [{}, {}) to [{}, {})",
                self.level,
                previous_span.start(),
                previous_span.end(),
                shared_span.start(),
                shared_span.end()
            )));
        }

        self.summaries = self.summaries.clone().slice_assign(
            [0..batch_size, last_index..last_index + 1, 0..summary_dim],
            node.summary().reshape([batch_size, 1, summary_dim]),
        );
        self.keys = self.keys.clone().slice_assign(
            [0..batch_size, last_index..last_index + 1, 0..key_dim],
            node.key().reshape([batch_size, 1, key_dim]),
        );
        self.values = self.values.clone().slice_assign(
            [0..batch_size, last_index..last_index + 1, 0..value_dim],
            node.value().reshape([batch_size, 1, value_dim]),
        );
        self.shared_spans[last_index] = shared_span;

        Ok(last_index)
    }
}

impl<B: Backend> Record<B> for TreeLevelStoreRecord<B> {
    type Item<S: PrecisionSettings> =
        TreeLevelStoreRecordItem<<Tensor<B, 3> as Record<B>>::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        TreeLevelStoreRecordItem {
            summaries: Record::<B>::into_item::<S>(self.summaries),
            keys: Record::<B>::into_item::<S>(self.keys),
            values: Record::<B>::into_item::<S>(self.values),
            level: self.level,
            shared_spans: self.shared_spans,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            summaries: Record::<B>::from_item::<S>(item.summaries, device),
            keys: Record::<B>::from_item::<S>(item.keys, device),
            values: Record::<B>::from_item::<S>(item.values, device),
            level: item.level,
            shared_spans: item.shared_spans,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeSummaryStateShape {
    pub levels: Vec<TreeLevelStoreShape>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeNodeAddress {
    level: usize,
    index: usize,
    shared_span: TokenSpan,
}

impl TreeNodeAddress {
    pub fn level(&self) -> usize {
        self.level
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn shared_span(&self) -> TokenSpan {
        self.shared_span
    }
}

#[derive(Debug, Clone)]
pub struct TreeSummaryStateRecord<B: Backend> {
    pub levels: Vec<TreeLevelStoreRecord<B>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSummaryStateRecordItem<TreeLevelItem> {
    pub levels: Vec<TreeLevelItem>,
}

#[derive(Debug, Clone)]
pub struct TreeSummaryState<B: Backend> {
    layout: FractalV2StateLayout,
    levels: Vec<TreeLevelStore<B>>,
}

impl<B: Backend> TreeSummaryState<B> {
    pub fn empty(layout: FractalV2StateLayout) -> Self {
        Self {
            layout,
            levels: Vec::new(),
        }
    }

    pub fn from_record(
        record: TreeSummaryStateRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        let mut state = Self::empty(layout);
        for level in record.levels {
            if level.level < state.levels.len() {
                return Err(FractalError::InvalidConfig(format!(
                    "tree_summary record contains duplicate level {}",
                    level.level
                )));
            }
            state.upsert_level(TreeLevelStore::from_record(level, layout)?)?;
        }
        if let Some(last_level) = state.levels.last() {
            if last_level.shared_spans().len() != 1 {
                return Err(FractalError::InvalidConfig(
                    "tree is missing one or more deterministic parent levels".to_string(),
                ));
            }
        }
        Ok(state)
    }

    pub fn into_record(&self) -> TreeSummaryStateRecord<B> {
        TreeSummaryStateRecord {
            levels: self
                .levels
                .iter()
                .map(TreeLevelStore::into_record)
                .collect(),
        }
    }

    pub fn shape(&self) -> TreeSummaryStateShape {
        TreeSummaryStateShape {
            levels: self
                .levels
                .iter()
                .map(|level| level.shape(self.layout.batch_timeline_mode))
                .collect(),
        }
    }

    pub fn levels(&self) -> &[TreeLevelStore<B>] {
        &self.levels
    }

    pub fn level(&self, level: usize) -> Option<&TreeLevelStore<B>> {
        self.levels.get(level)
    }

    pub fn key_dim(&self) -> usize {
        self.layout.key_dim
    }

    pub fn value_dim(&self) -> usize {
        self.layout.value_dim
    }

    pub fn root_address(&self) -> Option<TreeNodeAddress> {
        self.levels
            .last()
            .and_then(|level| self.node_address(level.level(), 0).ok())
    }

    pub fn node_address(
        &self,
        level: usize,
        index: usize,
    ) -> Result<TreeNodeAddress, FractalError> {
        let level_store = self.level(level).ok_or_else(|| {
            FractalError::InvalidState(format!("tree level {level} does not exist"))
        })?;
        let shared_span = level_store
            .shared_spans()
            .get(index)
            .copied()
            .ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "tree level {level} node index {index} is out of bounds for {} nodes",
                    level_store.node_count()
                ))
            })?;

        Ok(TreeNodeAddress {
            level,
            index,
            shared_span,
        })
    }

    pub fn child_addresses(
        &self,
        node: TreeNodeAddress,
    ) -> Result<Vec<TreeNodeAddress>, FractalError> {
        if node.level == 0 {
            return Ok(Vec::new());
        }

        let node = self.node_address(node.level, node.index)?;
        let child_level = self.level(node.level - 1).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "tree level {} does not exist while resolving children for level {} node {}",
                node.level - 1,
                node.level,
                node.index
            ))
        })?;
        let mut children = Vec::new();

        for (index, span) in child_level.shared_spans().iter().copied().enumerate() {
            if span.start() >= node.shared_span.start() && span.end() <= node.shared_span.end() {
                children.push(TreeNodeAddress {
                    level: node.level - 1,
                    index,
                    shared_span: span,
                });
            }
        }

        validate_child_partition(node, &children)?;
        Ok(children)
    }

    pub fn diagnostics(&self) -> TreeSummaryDiagnostics {
        TreeSummaryDiagnostics {
            nodes_per_level: self.levels.iter().map(TreeLevelStore::node_count).collect(),
            tree_depth_reached: self.levels.len(),
            has_dead_or_unused_nodes: validate_tree_level_prefix(self).is_err(),
        }
    }

    pub fn rebuild_from_sealed_leaves<TM: TreeMergeCell<B>>(
        &mut self,
        sealed_leaves: &LeafSummaryStore<B>,
        tree_merge_cell: &TM,
    ) -> Result<(), FractalError> {
        validate_tree_merge_shape(self.layout, tree_merge_cell.shape())?;

        if sealed_leaves.shared_spans().is_empty() {
            self.levels.clear();
            return Ok(());
        }

        let mut rebuilt_levels = Vec::new();
        let mut current_level = TreeLevelStore::from_parts(
            self.layout,
            0,
            sealed_leaves.summaries(),
            sealed_leaves.keys(),
            sealed_leaves.values(),
            sealed_leaves.shared_spans().to_vec(),
        )?;
        rebuilt_levels.push(current_level.clone());

        let mut level_index = 1usize;
        while current_level.node_count() > 1 {
            current_level =
                build_parent_level(&current_level, level_index, self.layout, tree_merge_cell)?;
            rebuilt_levels.push(current_level.clone());
            level_index += 1;
        }

        self.levels = rebuilt_levels;
        validate_tree_level_prefix(self)
    }

    pub fn append_sealed_leaf<TM: TreeMergeCell<B>>(
        &mut self,
        node: TreeNodeBatch<B>,
        shared_span: TokenSpan,
        tree_merge_cell: &TM,
    ) -> Result<(), FractalError> {
        validate_tree_merge_shape(self.layout, tree_merge_cell.shape())?;
        validate_token_span("tree.append_sealed_leaf.shared_span", shared_span)?;
        ensure_match(
            "tree.append_sealed_leaf.shared_span.len",
            shared_span.len(),
            self.layout.leaf_size,
        )?;

        if self.levels.is_empty() {
            let mut level0 = TreeLevelStore::empty(0, self.layout, &node.summary().device());
            level0.push_node(node, shared_span)?;
            self.levels.push(level0);
            return validate_tree_level_prefix(self);
        }

        self.levels[0].push_node(node, shared_span)?;
        let mut child_level_index = 0usize;
        loop {
            let child_level = &self.levels[child_level_index];
            let child_count = child_level.node_count();
            if child_count <= 1 {
                break;
            }

            let parent_level_index = child_level_index + 1;
            let (parent_node, parent_span) = if child_count.is_multiple_of(2) {
                let left_index = child_count - 2;
                let right_index = child_count - 1;
                let merged = tree_merge_cell
                    .merge_pair(
                        child_level.node(left_index)?,
                        child_level.node(right_index)?,
                        parent_level_index,
                    )?
                    .into_node()?;
                let span = TokenSpan::new(
                    child_level.shared_spans()[left_index].start(),
                    child_level.shared_spans()[right_index].end(),
                )?;
                (merged, span)
            } else {
                (
                    child_level.node(child_count - 1)?,
                    child_level.shared_spans()[child_count - 1],
                )
            };

            if parent_level_index == self.levels.len() {
                let mut parent_level = TreeLevelStore::empty(
                    parent_level_index,
                    self.layout,
                    &parent_node.summary().device(),
                );
                parent_level.push_node(parent_node, parent_span)?;
                self.levels.push(parent_level);
            } else {
                self.upsert_right_frontier(parent_level_index, parent_node, parent_span)?;
            }

            child_level_index = parent_level_index;
        }

        validate_tree_level_prefix(self)
    }

    fn upsert_right_frontier(
        &mut self,
        level_index: usize,
        node: TreeNodeBatch<B>,
        shared_span: TokenSpan,
    ) -> Result<(), FractalError> {
        let replace_last = self.levels[level_index]
            .shared_spans()
            .last()
            .is_some_and(|existing| existing.start() == shared_span.start());

        if replace_last {
            self.levels[level_index].replace_last_node(node, shared_span)?;
        } else {
            self.levels[level_index].push_node(node, shared_span)?;
        }

        Ok(())
    }

    pub fn upsert_level(&mut self, level_store: TreeLevelStore<B>) -> Result<(), FractalError> {
        let shape = level_store.shape(self.layout.batch_timeline_mode);
        ensure_match(
            "tree_level.batch_size",
            shape.batch_size,
            self.layout.batch_size,
        )?;
        ensure_match(
            "tree_level.summary_dim",
            shape.summary_dim,
            self.layout.summary_dim,
        )?;
        ensure_match("tree_level.key_dim", shape.key_dim, self.layout.key_dim)?;
        ensure_match(
            "tree_level.value_dim",
            shape.value_dim,
            self.layout.value_dim,
        )?;
        ensure_match(
            "tree_level.shared_span_count",
            level_store.shared_spans.len(),
            shape.node_count,
        )?;
        let level = shape.level;
        if level > self.levels.len() {
            return Err(FractalError::InvalidState(format!(
                "cannot insert tree level {level} before levels 0..{} exist",
                self.levels.len()
            )));
        }

        let previous_level = self.levels.get(level).cloned();
        if level == self.levels.len() {
            self.levels.push(level_store);
        } else {
            self.levels[level] = level_store;
        }

        if let Err(error) = validate_tree_level_prefix(self) {
            match previous_level {
                Some(previous_level) => self.levels[level] = previous_level,
                None => {
                    self.levels.pop();
                }
            }
            return Err(error);
        }

        Ok(())
    }
}

impl<B: Backend> Record<B> for TreeSummaryStateRecord<B> {
    type Item<S: PrecisionSettings> =
        TreeSummaryStateRecordItem<<TreeLevelStoreRecord<B> as Record<B>>::Item<S>>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        TreeSummaryStateRecordItem {
            levels: Record::<B>::into_item::<S>(self.levels),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            levels: Record::<B>::from_item::<S>(item.levels, device),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafTokenCacheShape {
    pub batch_size: usize,
    pub leaf_count: usize,
    pub tokens_per_leaf: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub batch_timeline_mode: BatchTimelineMode,
}

#[derive(Debug, Clone)]
pub struct LeafTokenCacheRecord<B: Backend> {
    pub keys: Tensor<B, 4>,
    pub values: Tensor<B, 4>,
    pub mask: Tensor<B, 3, Bool>,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeafTokenCacheRecordItem<Tensor4Item, MaskItem> {
    pub keys: Tensor4Item,
    pub values: Tensor4Item,
    pub mask: MaskItem,
    pub shared_spans: Vec<TokenSpan>,
}

#[derive(Debug, Clone)]
pub struct LeafTokenCache<B: Backend> {
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    mask: Tensor<B, 3, Bool>,
    shared_spans: Vec<TokenSpan>,
}

impl<B: Backend> LeafTokenCache<B> {
    pub fn empty(layout: FractalV2StateLayout, device: &B::Device) -> Self {
        Self {
            keys: Tensor::<B, 4>::zeros(
                [
                    layout.batch_size,
                    0,
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                device,
            ),
            values: Tensor::<B, 4>::zeros(
                [
                    layout.batch_size,
                    0,
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                device,
            ),
            mask: false_mask([layout.batch_size, 0, layout.leaf_size], device),
            shared_spans: Vec::new(),
        }
    }

    pub fn from_record(
        record: LeafTokenCacheRecord<B>,
        layout: FractalV2StateLayout,
    ) -> Result<Self, FractalError> {
        let key_dims = record.keys.dims();
        let leaf_count = key_dims[1];
        ensure_dims4(
            "leaf_token_cache.keys",
            key_dims,
            [
                layout.batch_size,
                leaf_count,
                layout.leaf_size,
                layout.token_cache_key_dim,
            ],
        )?;
        ensure_dims4(
            "leaf_token_cache.values",
            record.values.dims(),
            [
                layout.batch_size,
                leaf_count,
                layout.leaf_size,
                layout.token_cache_value_dim,
            ],
        )?;
        ensure_dims3(
            "leaf_token_cache.mask",
            record.mask.dims(),
            [layout.batch_size, leaf_count, layout.leaf_size],
        )?;
        validate_fixed_width_prefix_spans(
            "leaf_token_cache.shared_spans",
            &record.shared_spans,
            layout.leaf_size,
        )?;
        ensure_match(
            "leaf_token_cache.shared_span_count",
            record.shared_spans.len(),
            leaf_count,
        )?;
        let expected_true_count = checked_usize_product(
            "leaf_token_cache.mask_true_count",
            &[layout.batch_size, leaf_count, layout.leaf_size],
        )?;
        ensure_match(
            "leaf_token_cache.mask_true_count",
            count_true(record.mask.clone()),
            expected_true_count,
        )?;

        Ok(Self {
            keys: record.keys,
            values: record.values,
            mask: record.mask,
            shared_spans: record.shared_spans,
        })
    }

    pub fn into_record(&self) -> LeafTokenCacheRecord<B> {
        LeafTokenCacheRecord {
            keys: self.keys.clone(),
            values: self.values.clone(),
            mask: self.mask.clone(),
            shared_spans: self.shared_spans.clone(),
        }
    }

    pub fn shape(&self, batch_timeline_mode: BatchTimelineMode) -> LeafTokenCacheShape {
        let [batch_size, leaf_count, tokens_per_leaf, key_dim] = self.keys.dims();
        let [_, _, _, value_dim] = self.values.dims();
        LeafTokenCacheShape {
            batch_size,
            leaf_count,
            tokens_per_leaf,
            key_dim,
            value_dim,
            batch_timeline_mode,
        }
    }

    pub fn keys(&self) -> Tensor<B, 4> {
        self.keys.clone()
    }

    pub fn values(&self) -> Tensor<B, 4> {
        self.values.clone()
    }

    pub fn mask(&self) -> Tensor<B, 3, Bool> {
        self.mask.clone()
    }

    pub fn shared_spans(&self) -> &[TokenSpan] {
        &self.shared_spans
    }

    pub(crate) fn push_sealed_leaf(
        &mut self,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
        mask: Tensor<B, 2, Bool>,
        tokens_per_leaf_limit: usize,
        shared_span: TokenSpan,
    ) -> Result<usize, FractalError> {
        let [batch_size, tokens_per_leaf, key_dim] = keys.dims();
        let [value_batch_size, value_tokens_per_leaf, value_dim] = values.dims();
        let [mask_batch_size, mask_tokens_per_leaf] = mask.dims();
        let [expected_batch_size, leaf_count, expected_tokens_per_leaf, expected_key_dim] =
            self.keys.dims();
        let [_, _, _, expected_value_dim] = self.values.dims();
        ensure_match(
            "leaf_token_cache.push.batch_size",
            batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_token_cache.push.value_batch_size",
            value_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_token_cache.push.mask_batch_size",
            mask_batch_size,
            expected_batch_size,
        )?;
        ensure_match(
            "leaf_token_cache.push.tokens_per_leaf",
            tokens_per_leaf,
            expected_tokens_per_leaf,
        )?;
        ensure_match(
            "leaf_token_cache.push.value_tokens_per_leaf",
            value_tokens_per_leaf,
            expected_tokens_per_leaf,
        )?;
        ensure_match(
            "leaf_token_cache.push.mask_tokens_per_leaf",
            mask_tokens_per_leaf,
            expected_tokens_per_leaf,
        )?;
        ensure_match("leaf_token_cache.push.key_dim", key_dim, expected_key_dim)?;
        ensure_match(
            "leaf_token_cache.push.value_dim",
            value_dim,
            expected_value_dim,
        )?;
        ensure_nonzero(
            "leaf_token_cache.push.tokens_per_leaf_limit",
            tokens_per_leaf_limit,
        )?;
        ensure_match(
            "leaf_token_cache.push.tokens_per_leaf_limit",
            tokens_per_leaf_limit,
            expected_tokens_per_leaf,
        )?;
        ensure_match(
            "leaf_token_cache.push.shared_span.len",
            shared_span.len(),
            tokens_per_leaf_limit,
        )?;
        let expected_true_count = checked_usize_product(
            "leaf_token_cache.push.mask_true_count",
            &[expected_batch_size, expected_tokens_per_leaf],
        )?;
        ensure_match(
            "leaf_token_cache.push.mask_true_count",
            count_true(mask.clone()),
            expected_true_count,
        )?;

        let leaf_index = leaf_count;
        let expected_span = TokenSpan::new(
            leaf_index
                .checked_mul(tokens_per_leaf_limit)
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "leaf token cache index overflow while computing expected span".to_string(),
                    )
                })?,
            (leaf_index + 1)
                .checked_mul(tokens_per_leaf_limit)
                .ok_or_else(|| {
                    FractalError::InvalidState(
                        "leaf token cache index overflow while computing expected span".to_string(),
                    )
                })?,
        )?;
        if shared_span != expected_span {
            return Err(FractalError::InvalidState(format!(
                "leaf token cache span mismatch: expected [{}, {}), got [{}, {})",
                expected_span.start(),
                expected_span.end(),
                shared_span.start(),
                shared_span.end()
            )));
        }

        self.keys = Tensor::cat(
            vec![
                self.keys.clone(),
                keys.reshape([batch_size, 1, tokens_per_leaf, key_dim]),
            ],
            1,
        );
        self.values = Tensor::cat(
            vec![
                self.values.clone(),
                values.reshape([batch_size, 1, tokens_per_leaf, value_dim]),
            ],
            1,
        );
        self.mask = Tensor::cat(
            vec![
                self.mask.clone(),
                mask.reshape([batch_size, 1, tokens_per_leaf]),
            ],
            1,
        );
        self.shared_spans.push(shared_span);

        Ok(leaf_index)
    }
}

impl<B: Backend> Record<B> for LeafTokenCacheRecord<B> {
    type Item<S: PrecisionSettings> = LeafTokenCacheRecordItem<
        <Tensor<B, 4> as Record<B>>::Item<S>,
        <Tensor<B, 3, Bool> as Record<B>>::Item<S>,
    >;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        LeafTokenCacheRecordItem {
            keys: Record::<B>::into_item::<S>(self.keys),
            values: Record::<B>::into_item::<S>(self.values),
            mask: Record::<B>::into_item::<S>(self.mask),
            shared_spans: self.shared_spans,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            keys: Record::<B>::from_item::<S>(item.keys, device),
            values: Record::<B>::from_item::<S>(item.values, device),
            mask: Record::<B>::from_item::<S>(item.mask, device),
            shared_spans: item.shared_spans,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FractalV2StateShape {
    pub layout: FractalV2StateLayout,
    pub roots: MultiRootStateShape,
    pub live_leaf: LiveLeafStateShape,
    pub sealed_leaves: LeafSummaryStoreShape,
    pub tree: TreeSummaryStateShape,
    pub leaf_token_cache: LeafTokenCacheShape,
    pub retrieval_policy: RetrievalPolicy,
    pub merge_policy: MergeCheckpointPolicy,
}

#[derive(Debug, Clone)]
pub struct FractalV2StateRecord<B: Backend> {
    pub layout: FractalV2StateLayout,
    pub roots: MultiRootStateRecord<B>,
    pub live_leaf: LiveLeafStateRecord<B>,
    pub sealed_leaves: LeafSummaryStoreRecord<B>,
    pub tree: TreeSummaryStateRecord<B>,
    pub leaf_token_cache: LeafTokenCacheRecord<B>,
    pub retrieval_policy: RetrievalPolicy,
    pub merge_policy: MergeCheckpointPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalV2StateRecordItem<RootsItem, LiveLeafItem, LeafSummaryItem, TreeItem, CacheItem> {
    pub layout: FractalV2StateLayout,
    pub roots: RootsItem,
    pub live_leaf: LiveLeafItem,
    pub sealed_leaves: LeafSummaryItem,
    pub tree: TreeItem,
    pub leaf_token_cache: CacheItem,
    pub retrieval_policy: RetrievalPolicy,
    pub merge_policy: MergeCheckpointPolicy,
}

#[derive(Debug, Clone)]
pub struct FractalV2State<B: Backend> {
    layout: FractalV2StateLayout,
    roots: MultiRootState<B>,
    live_leaf: LiveLeafState<B>,
    sealed_leaves: LeafSummaryStore<B>,
    tree: TreeSummaryState<B>,
    leaf_token_cache: LeafTokenCache<B>,
    retrieval_policy: RetrievalPolicy,
    merge_policy: MergeCheckpointPolicy,
}

impl<B: Backend> FractalV2State<B> {
    pub fn for_model_shape(
        model_shape: FractalV2ModelShape,
        batch_size: usize,
        merge_policy: MergeCheckpointPolicy,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        let model_shape = model_shape.validate()?;
        let layout = FractalV2StateLayout::from_model_shape(model_shape, batch_size)?;
        let retrieval_policy = RetrievalPolicy::from_router_shape(model_shape.router)?;
        Self::new(layout, retrieval_policy, merge_policy, device)
    }

    pub fn from_record(
        record: FractalV2StateRecord<B>,
        model_shape: FractalV2ModelShape,
    ) -> Result<Self, FractalError> {
        let model_shape = model_shape.validate()?;
        let expected_layout =
            FractalV2StateLayout::from_model_shape(model_shape, record.layout.batch_size)?;
        if record.layout != expected_layout {
            return Err(FractalError::InvalidConfig(format!(
                "state_record.layout mismatch: expected {:?}, got {:?}",
                expected_layout, record.layout
            )));
        }

        let expected_retrieval_policy = RetrievalPolicy::from_router_shape(model_shape.router)?;
        if record.retrieval_policy != expected_retrieval_policy {
            return Err(FractalError::InvalidConfig(format!(
                "state_record.retrieval_policy mismatch: expected {:?}, got {:?}",
                expected_retrieval_policy, record.retrieval_policy
            )));
        }

        record.merge_policy.validate()?;
        ensure_match(
            "merge_checkpoint_policy.tokens_per_leaf",
            record.merge_policy.tokens_per_leaf(),
            record.layout.leaf_size,
        )?;

        let roots = MultiRootState::from_record(record.roots, record.layout)?;
        let live_leaf = LiveLeafState::from_record(record.live_leaf, record.layout)?;
        let sealed_leaves = LeafSummaryStore::from_record(record.sealed_leaves, record.layout)?;
        let tree = TreeSummaryState::from_record(record.tree, record.layout)?;
        let leaf_token_cache = LeafTokenCache::from_record(record.leaf_token_cache, record.layout)?;

        validate_state_consistency(
            record.layout,
            &live_leaf,
            &sealed_leaves,
            &tree,
            &leaf_token_cache,
        )?;

        Ok(Self {
            layout: record.layout,
            roots,
            live_leaf,
            sealed_leaves,
            tree,
            leaf_token_cache,
            retrieval_policy: record.retrieval_policy,
            merge_policy: record.merge_policy,
        })
    }

    pub fn into_record(&self) -> FractalV2StateRecord<B> {
        FractalV2StateRecord {
            layout: self.layout,
            roots: self.roots.into_record(),
            live_leaf: self.live_leaf.into_record(),
            sealed_leaves: self.sealed_leaves.into_record(),
            tree: self.tree.into_record(),
            leaf_token_cache: self.leaf_token_cache.into_record(),
            retrieval_policy: self.retrieval_policy,
            merge_policy: self.merge_policy,
        }
    }

    fn new(
        layout: FractalV2StateLayout,
        retrieval_policy: RetrievalPolicy,
        merge_policy: MergeCheckpointPolicy,
        device: &B::Device,
    ) -> Result<Self, FractalError> {
        layout.validate()?;
        retrieval_policy.validate()?;
        merge_policy.validate()?;
        ensure_match(
            "merge_checkpoint_policy.tokens_per_leaf",
            merge_policy.tokens_per_leaf(),
            layout.leaf_size,
        )?;

        let state = Self {
            layout,
            roots: MultiRootState::zeros(layout, device),
            live_leaf: LiveLeafState::empty(layout, device),
            sealed_leaves: LeafSummaryStore::empty(layout, device),
            tree: TreeSummaryState::empty(layout),
            leaf_token_cache: LeafTokenCache::empty(layout, device),
            retrieval_policy,
            merge_policy,
        };
        validate_state_consistency(
            state.layout,
            &state.live_leaf,
            &state.sealed_leaves,
            &state.tree,
            &state.leaf_token_cache,
        )?;
        Ok(state)
    }

    pub fn layout(&self) -> FractalV2StateLayout {
        self.layout
    }

    pub fn roots(&self) -> &MultiRootState<B> {
        &self.roots
    }

    pub fn update_roots(&mut self, roots: MultiRootState<B>) -> Result<(), FractalError> {
        let shape = roots.shape();
        ensure_match(
            "state.update_roots.batch_size",
            shape.batch_size,
            self.layout.batch_size,
        )?;
        ensure_match(
            "state.update_roots.root_count",
            shape.root_count,
            self.layout.root_count,
        )?;
        ensure_match(
            "state.update_roots.recurrent_dim",
            shape.recurrent_dim,
            self.layout.root_state_dim,
        )?;
        ensure_match(
            "state.update_roots.intent_dim",
            shape.intent_dim,
            self.layout.root_readout_dim,
        )?;
        self.roots = roots;
        Ok(())
    }

    pub fn live_leaf(&self) -> &LiveLeafState<B> {
        &self.live_leaf
    }

    pub fn sealed_leaves(&self) -> &LeafSummaryStore<B> {
        &self.sealed_leaves
    }

    pub fn tree(&self) -> &TreeSummaryState<B> {
        &self.tree
    }

    pub fn leaf_token_cache(&self) -> &LeafTokenCache<B> {
        &self.leaf_token_cache
    }

    pub fn retrieval_policy(&self) -> RetrievalPolicy {
        self.retrieval_policy
    }

    pub fn merge_policy(&self) -> MergeCheckpointPolicy {
        self.merge_policy
    }

    pub fn append_root_readouts<LS: LeafSummarizer<B>, TM: TreeMergeCell<B>>(
        &mut self,
        root_readouts: Tensor<B, 3>,
        leaf_summarizer: &LS,
        tree_merge_cell: &TM,
    ) -> Result<Option<SealedLeafMaterialization<B>>, FractalError> {
        self.append_root_readouts_with_active_root_count(
            root_readouts,
            self.layout.root_count,
            leaf_summarizer,
            tree_merge_cell,
        )
    }

    pub fn append_root_readouts_with_active_root_count<
        LS: LeafSummarizer<B>,
        TM: TreeMergeCell<B>,
    >(
        &mut self,
        root_readouts: Tensor<B, 3>,
        active_root_count: usize,
        leaf_summarizer: &LS,
        tree_merge_cell: &TM,
    ) -> Result<Option<SealedLeafMaterialization<B>>, FractalError> {
        let [batch_size, root_count, readout_dim] = root_readouts.dims();
        ensure_match(
            "state.append_root_readouts.batch_size",
            batch_size,
            self.layout.batch_size,
        )?;
        ensure_match(
            "state.append_root_readouts.root_count",
            root_count,
            self.layout.root_count,
        )?;
        ensure_match(
            "state.append_root_readouts.readout_dim",
            readout_dim,
            self.layout.root_readout_dim,
        )?;
        if active_root_count == 0 || active_root_count > root_count {
            return Err(FractalError::InvalidConfig(format!(
                "state.append_root_readouts.active_root_count must be within 1..={root_count}, got {active_root_count}"
            )));
        }
        let summarizer_shape = leaf_summarizer.shape();
        ensure_match(
            "state.append_root_readouts.summarizer.readout_dim",
            summarizer_shape.readout_dim,
            self.layout.root_readout_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.leaf_size",
            summarizer_shape.leaf_size,
            self.layout.leaf_size,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.summary_dim",
            summarizer_shape.summary_dim,
            self.layout.summary_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.key_dim",
            summarizer_shape.key_dim,
            self.layout.key_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.value_dim",
            summarizer_shape.value_dim,
            self.layout.value_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.token_cache_key_dim",
            summarizer_shape.token_cache_key_dim,
            self.layout.token_cache_key_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.summarizer.token_cache_value_dim",
            summarizer_shape.token_cache_value_dim,
            self.layout.token_cache_value_dim,
        )?;
        let tree_merge_shape = tree_merge_cell.shape();
        ensure_match(
            "state.append_root_readouts.tree_merge_cell.summary_dim",
            tree_merge_shape.summary_dim,
            self.layout.summary_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.tree_merge_cell.key_dim",
            tree_merge_shape.key_dim,
            self.layout.key_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.tree_merge_cell.value_dim",
            tree_merge_shape.value_dim,
            self.layout.value_dim,
        )?;
        ensure_match(
            "state.append_root_readouts.tree_merge_cell.scale_embedding_dim",
            tree_merge_shape.scale_embedding_dim,
            self.layout.scale_embedding_dim,
        )?;

        let Some((sealed_token_readouts, shared_span)) =
            self.live_leaf.append_root_readouts(root_readouts)?
        else {
            validate_state_consistency(
                self.layout,
                &self.live_leaf,
                &self.sealed_leaves,
                &self.tree,
                &self.leaf_token_cache,
            )?;
            return Ok(None);
        };
        let sealed_token_readouts = if active_root_count == root_count {
            sealed_token_readouts
        } else {
            sealed_token_readouts.narrow(1, 0, active_root_count)
        };

        let (summary, key, value, token_keys, token_values, token_mask) = leaf_summarizer
            .summarize_sealed_leaf(sealed_token_readouts)?
            .into_parts();
        let leaf_index = self.sealed_leaves.push_sealed_leaf(
            summary.clone(),
            key.clone(),
            value.clone(),
            self.layout.leaf_size,
            shared_span,
        )?;
        let cache_leaf_index = self.leaf_token_cache.push_sealed_leaf(
            token_keys.clone(),
            token_values.clone(),
            token_mask.clone(),
            self.layout.leaf_size,
            shared_span,
        )?;
        ensure_match(
            "state.append_root_readouts.cache_leaf_index",
            cache_leaf_index,
            leaf_index,
        )?;
        self.tree.append_sealed_leaf(
            TreeNodeBatch::from_tensors(summary.clone(), key.clone(), value.clone())?,
            shared_span,
            tree_merge_cell,
        )?;
        validate_state_consistency(
            self.layout,
            &self.live_leaf,
            &self.sealed_leaves,
            &self.tree,
            &self.leaf_token_cache,
        )?;

        Ok(Some(SealedLeafMaterialization {
            leaf_index,
            shared_span,
            summary,
            key,
            value,
            token_keys,
            token_values,
            token_mask,
        }))
    }

    pub fn shape(&self) -> FractalV2StateShape {
        FractalV2StateShape {
            layout: self.layout,
            roots: self.roots.shape(),
            live_leaf: self.live_leaf.shape(self.layout.batch_timeline_mode),
            sealed_leaves: self.sealed_leaves.shape(self.layout.batch_timeline_mode),
            tree: self.tree.shape(),
            leaf_token_cache: self.leaf_token_cache.shape(self.layout.batch_timeline_mode),
            retrieval_policy: self.retrieval_policy,
            merge_policy: self.merge_policy,
        }
    }
}

impl<B: Backend> Record<B> for FractalV2StateRecord<B> {
    type Item<S: PrecisionSettings> = FractalV2StateRecordItem<
        <MultiRootStateRecord<B> as Record<B>>::Item<S>,
        <LiveLeafStateRecord<B> as Record<B>>::Item<S>,
        <LeafSummaryStoreRecord<B> as Record<B>>::Item<S>,
        <TreeSummaryStateRecord<B> as Record<B>>::Item<S>,
        <LeafTokenCacheRecord<B> as Record<B>>::Item<S>,
    >;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        FractalV2StateRecordItem {
            layout: self.layout,
            roots: Record::<B>::into_item::<S>(self.roots),
            live_leaf: Record::<B>::into_item::<S>(self.live_leaf),
            sealed_leaves: Record::<B>::into_item::<S>(self.sealed_leaves),
            tree: Record::<B>::into_item::<S>(self.tree),
            leaf_token_cache: Record::<B>::into_item::<S>(self.leaf_token_cache),
            retrieval_policy: self.retrieval_policy,
            merge_policy: self.merge_policy,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            layout: item.layout,
            roots: Record::<B>::from_item::<S>(item.roots, device),
            live_leaf: Record::<B>::from_item::<S>(item.live_leaf, device),
            sealed_leaves: Record::<B>::from_item::<S>(item.sealed_leaves, device),
            tree: Record::<B>::from_item::<S>(item.tree, device),
            leaf_token_cache: Record::<B>::from_item::<S>(item.leaf_token_cache, device),
            retrieval_policy: item.retrieval_policy,
            merge_policy: item.merge_policy,
        }
    }
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

fn ensure_at_most(name: &str, actual: usize, limit: usize) -> Result<(), FractalError> {
    if actual > limit {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be at most {limit}, got {actual}"
        )));
    }

    Ok(())
}

fn ensure_multiple_of(name: &str, actual: usize, base: usize) -> Result<(), FractalError> {
    if !actual.is_multiple_of(base) {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be a multiple of {base}, got {actual}"
        )));
    }

    Ok(())
}

fn ensure_dims3(name: &str, actual: [usize; 3], expected: [usize; 3]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} dims mismatch: expected {:?}, got {:?}",
            expected, actual
        )));
    }

    Ok(())
}

fn ensure_dims4(name: &str, actual: [usize; 4], expected: [usize; 4]) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} dims mismatch: expected {:?}, got {:?}",
            expected, actual
        )));
    }

    Ok(())
}

fn validate_token_span(name: &str, span: TokenSpan) -> Result<(), FractalError> {
    TokenSpan::new(span.start(), span.end())
        .map(|_| ())
        .map_err(|_| {
            FractalError::InvalidConfig(format!(
                "{name} is invalid: start={} end={}",
                span.start(),
                span.end()
            ))
        })
}

fn validate_fixed_width_prefix_spans(
    name: &str,
    spans: &[TokenSpan],
    width: usize,
) -> Result<(), FractalError> {
    for (index, span) in spans.iter().copied().enumerate() {
        validate_token_span(&format!("{name}[{index}]"), span)?;
        let expected_start = index.checked_mul(width).ok_or_else(|| {
            FractalError::InvalidConfig(format!("{name}[{index}] start overflow for width {width}"))
        })?;
        let expected_end = expected_start.checked_add(width).ok_or_else(|| {
            FractalError::InvalidConfig(format!("{name}[{index}] end overflow for width {width}"))
        })?;
        if span.start() != expected_start || span.end() != expected_end {
            return Err(FractalError::InvalidConfig(format!(
                "{name}[{index}] mismatch: expected [{expected_start}, {expected_end}), got [{}, {})",
                span.start(),
                span.end()
            )));
        }
    }

    Ok(())
}

fn validate_contiguous_prefix_spans(name: &str, spans: &[TokenSpan]) -> Result<(), FractalError> {
    let mut expected_start = 0usize;
    for (index, span) in spans.iter().copied().enumerate() {
        validate_token_span(&format!("{name}[{index}]"), span)?;
        if span.start() != expected_start {
            return Err(FractalError::InvalidConfig(format!(
                "{name}[{index}] mismatch: expected start {expected_start}, got {}",
                span.start()
            )));
        }
        expected_start = span.end();
    }

    Ok(())
}

fn next_tree_level_spans(spans: &[TokenSpan]) -> Result<Vec<TokenSpan>, FractalError> {
    let mut next_level_spans = Vec::new();
    for pair in spans.chunks(2) {
        let merged = if pair.len() == 2 {
            TokenSpan::new(pair[0].start(), pair[1].end())?
        } else {
            pair[0]
        };
        next_level_spans.push(merged);
    }

    Ok(next_level_spans)
}

fn build_parent_level<B: Backend, TM: TreeMergeCell<B>>(
    current_level: &TreeLevelStore<B>,
    parent_level_index: usize,
    layout: FractalV2StateLayout,
    tree_merge_cell: &TM,
) -> Result<TreeLevelStore<B>, FractalError> {
    if current_level.node_count() <= 1 {
        return Err(FractalError::InvalidState(format!(
            "tree level {} cannot build a parent from {} node(s)",
            current_level.level(),
            current_level.node_count()
        )));
    }

    let mut summaries = Vec::with_capacity(current_level.node_count().div_ceil(2));
    let mut keys = Vec::with_capacity(current_level.node_count().div_ceil(2));
    let mut values = Vec::with_capacity(current_level.node_count().div_ceil(2));
    let mut shared_spans = Vec::with_capacity(current_level.node_count().div_ceil(2));

    for pair_start in (0..current_level.node_count()).step_by(2) {
        if pair_start + 1 < current_level.node_count() {
            let left = current_level.node(pair_start)?;
            let right = current_level.node(pair_start + 1)?;
            let merged = tree_merge_cell.merge_pair(left, right, parent_level_index)?;
            let merged_span = TokenSpan::new(
                current_level.shared_spans()[pair_start].start(),
                current_level.shared_spans()[pair_start + 1].end(),
            )?;
            summaries.push(merged.summary());
            keys.push(merged.key());
            values.push(merged.value());
            shared_spans.push(merged_span);
        } else {
            let carried = current_level.node(pair_start)?;
            summaries.push(carried.summary());
            keys.push(carried.key());
            values.push(carried.value());
            shared_spans.push(current_level.shared_spans()[pair_start]);
        }
    }

    let device = current_level.summaries().device();
    TreeLevelStore::from_parts(
        layout,
        parent_level_index,
        stack_rank2_batches(
            summaries,
            layout.batch_size,
            layout.summary_dim,
            &device,
            "tree_parent.summaries",
        )?,
        stack_rank2_batches(
            keys,
            layout.batch_size,
            layout.key_dim,
            &device,
            "tree_parent.keys",
        )?,
        stack_rank2_batches(
            values,
            layout.batch_size,
            layout.value_dim,
            &device,
            "tree_parent.values",
        )?,
        shared_spans,
    )
}

fn stack_rank2_batches<B: Backend>(
    batches: Vec<Tensor<B, 2>>,
    batch_size: usize,
    feature_dim: usize,
    device: &B::Device,
    name: &str,
) -> Result<Tensor<B, 3>, FractalError> {
    if batches.is_empty() {
        return Ok(Tensor::<B, 3>::zeros([batch_size, 0, feature_dim], device));
    }

    let mut reshaped = Vec::with_capacity(batches.len());
    for batch in batches {
        let [actual_batch_size, actual_feature_dim] = batch.dims();
        ensure_match(&format!("{name}.batch_size"), actual_batch_size, batch_size)?;
        ensure_match(
            &format!("{name}.feature_dim"),
            actual_feature_dim,
            feature_dim,
        )?;
        reshaped.push(batch.reshape([batch_size, 1, feature_dim]));
    }

    Ok(Tensor::cat(reshaped, 1))
}

fn validate_tree_merge_shape(
    layout: FractalV2StateLayout,
    merge_shape: crate::v2::TreeMergeCellShape,
) -> Result<(), FractalError> {
    ensure_match(
        "tree_merge_cell.summary_dim",
        merge_shape.summary_dim,
        layout.summary_dim,
    )?;
    ensure_match(
        "tree_merge_cell.key_dim",
        merge_shape.key_dim,
        layout.key_dim,
    )?;
    ensure_match(
        "tree_merge_cell.value_dim",
        merge_shape.value_dim,
        layout.value_dim,
    )?;
    ensure_match(
        "tree_merge_cell.scale_embedding_dim",
        merge_shape.scale_embedding_dim,
        layout.scale_embedding_dim,
    )
}

fn validate_tree_level_prefix<B: Backend>(tree: &TreeSummaryState<B>) -> Result<(), FractalError> {
    if tree.levels().is_empty() {
        return Ok(());
    }

    let mut expected_level_spans = tree.levels()[0].shared_spans().to_vec();
    if expected_level_spans.is_empty() {
        return Err(FractalError::InvalidConfig(
            "tree must not contain an empty level 0 frontier".to_string(),
        ));
    }
    validate_fixed_width_prefix_spans(
        "tree.levels[0].shared_spans",
        &expected_level_spans,
        tree.layout.leaf_size,
    )?;

    for (level_index, level) in tree.levels().iter().enumerate() {
        validate_contiguous_prefix_spans(
            &format!("tree.levels[{level_index}].shared_spans"),
            level.shared_spans(),
        )?;
        if level.shared_spans() != expected_level_spans.as_slice() {
            return Err(FractalError::InvalidConfig(format!(
                "tree level {level_index} spans do not match the expected dyadic parent chain"
            )));
        }

        if level_index + 1 < tree.levels().len() {
            if expected_level_spans.len() == 1 {
                return Err(FractalError::InvalidConfig(format!(
                    "tree contains trailing level {} beyond the canonical root",
                    level_index + 1
                )));
            }
            expected_level_spans = next_tree_level_spans(&expected_level_spans)?;
        }
    }

    Ok(())
}

fn validate_tree_parent_chain<B: Backend>(
    tree: &TreeSummaryState<B>,
    sealed_spans: &[TokenSpan],
) -> Result<(), FractalError> {
    if sealed_spans.is_empty() {
        if !tree.levels().is_empty() {
            return Err(FractalError::InvalidConfig(
                "tree must be empty when there are no sealed leaves".to_string(),
            ));
        }
        return Ok(());
    }

    if tree.levels().is_empty() {
        return Err(FractalError::InvalidConfig(
            "tree must include level 0 once sealed leaves exist".to_string(),
        ));
    }

    if tree.levels()[0].shared_spans() != sealed_spans {
        return Err(FractalError::InvalidConfig(
            "tree level 0 spans must match sealed leaf spans".to_string(),
        ));
    }

    let mut expected_level_spans = sealed_spans.to_vec();
    for (level_index, level) in tree.levels().iter().enumerate() {
        if expected_level_spans.is_empty() {
            return Err(FractalError::InvalidConfig(format!(
                "tree contains trailing empty level {level_index}"
            )));
        }
        validate_contiguous_prefix_spans(
            &format!("tree.levels[{level_index}].shared_spans"),
            level.shared_spans(),
        )?;
        if level.shared_spans() != expected_level_spans.as_slice() {
            return Err(FractalError::InvalidConfig(format!(
                "tree level {level_index} spans do not match the expected dyadic parent chain"
            )));
        }

        if expected_level_spans.len() == 1 {
            expected_level_spans.clear();
            continue;
        }

        expected_level_spans = next_tree_level_spans(&expected_level_spans)?;
    }

    if !expected_level_spans.is_empty() {
        return Err(FractalError::InvalidConfig(
            "tree is missing one or more deterministic parent levels".to_string(),
        ));
    }

    Ok(())
}

fn checked_usize_product(name: &str, factors: &[usize]) -> Result<usize, FractalError> {
    let mut product = 1usize;
    for factor in factors {
        product = product.checked_mul(*factor).ok_or_else(|| {
            FractalError::InvalidConfig(format!("{name} overflow while multiplying by {factor}"))
        })?;
    }

    Ok(product)
}

fn validate_state_consistency<B: Backend>(
    layout: FractalV2StateLayout,
    live_leaf: &LiveLeafState<B>,
    sealed_leaves: &LeafSummaryStore<B>,
    tree: &TreeSummaryState<B>,
    leaf_token_cache: &LeafTokenCache<B>,
) -> Result<(), FractalError> {
    validate_token_span("live_leaf.shared_span", live_leaf.shared_span())?;
    ensure_at_most(
        "live_leaf.shared_valid_tokens",
        live_leaf.shared_valid_tokens(),
        layout.leaf_size,
    )?;
    ensure_match(
        "live_leaf.shared_valid_tokens",
        live_leaf.shared_valid_tokens(),
        live_leaf.shared_span().len(),
    )?;
    ensure_multiple_of(
        "live_leaf.shared_span.start",
        live_leaf.shared_span().start(),
        layout.leaf_size,
    )?;
    let expected_live_leaf_start = checked_usize_product(
        "live_leaf.shared_span.start",
        &[sealed_leaves.shared_spans().len(), layout.leaf_size],
    )?;
    ensure_match(
        "live_leaf.shared_span.start",
        live_leaf.shared_span().start(),
        expected_live_leaf_start,
    )?;
    validate_fixed_width_prefix_spans(
        "sealed_leaves.shared_spans",
        sealed_leaves.shared_spans(),
        layout.leaf_size,
    )?;
    validate_fixed_width_prefix_spans(
        "leaf_token_cache.shared_spans",
        leaf_token_cache.shared_spans(),
        layout.leaf_size,
    )?;
    if sealed_leaves.shared_spans() != leaf_token_cache.shared_spans() {
        return Err(FractalError::InvalidConfig(
            "sealed leaf summaries and leaf token cache must describe the same sealed leaves"
                .to_string(),
        ));
    }

    validate_tree_parent_chain(tree, sealed_leaves.shared_spans())
}

fn validate_child_partition(
    parent: TreeNodeAddress,
    children: &[TreeNodeAddress],
) -> Result<(), FractalError> {
    if children.is_empty() {
        return Err(FractalError::InvalidState(format!(
            "tree level {} node {} has no child nodes in level {}",
            parent.level(),
            parent.index(),
            parent.level().saturating_sub(1)
        )));
    }

    ensure_match(
        "tree_node.children.first_start",
        children[0].shared_span().start(),
        parent.shared_span().start(),
    )?;
    ensure_match(
        "tree_node.children.last_end",
        children[children.len() - 1].shared_span().end(),
        parent.shared_span().end(),
    )?;

    for window in children.windows(2) {
        ensure_match(
            "tree_node.children.contiguous_spans",
            window[0].shared_span().end(),
            window[1].shared_span().start(),
        )?;
    }

    Ok(())
}

fn validate_live_leaf_zero_tail<B: Backend>(
    token_readouts: &Tensor<B, 4>,
    shared_valid_tokens: usize,
) -> Result<(), FractalError> {
    let [batch_size, root_count, leaf_size, readout_dim] = token_readouts.dims();
    if shared_valid_tokens >= leaf_size {
        return Ok(());
    }

    let values = token_readouts
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|error| {
            FractalError::InvalidConfig(format!(
                "live_leaf.token_readouts could not be inspected for causal tail zeros: {error}"
            ))
        })?;

    for batch_index in 0..batch_size {
        for root_index in 0..root_count {
            for token_index in shared_valid_tokens..leaf_size {
                for readout_index in 0..readout_dim {
                    let flat_index = (((batch_index * root_count + root_index) * leaf_size
                        + token_index)
                        * readout_dim)
                        + readout_index;
                    if values[flat_index] != 0.0 {
                        return Err(FractalError::InvalidConfig(format!(
                            "live_leaf.token_readouts must be zero beyond shared_valid_tokens; found nonzero value at batch {batch_index}, root {root_index}, token {token_index}, readout {readout_index}"
                        )));
                    }
                }
            }
        }
    }

    Ok(())
}

fn false_mask<B: Backend, const D: usize>(
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D, Bool> {
    Tensor::<B, D, Int>::zeros(shape, device).greater_elem(0)
}

fn count_true<B: Backend, const D: usize>(mask: Tensor<B, D, Bool>) -> usize {
    mask.int().sum().into_scalar().elem::<i64>() as usize
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Candle,
        module::Module,
        record::{FullPrecisionSettings, Record},
        tensor::TensorData,
    };
    use core::marker::PhantomData;

    use super::*;
    use crate::v2::{
        BaselineLeafSummarizer, BaselineLeafSummarizerConfig, BaselineTreeMergeCell,
        BaselineTreeMergeCellConfig, ExactLeafReadShape, FractalRouterHeadShape,
        FractalV2ModelShape, LeafSummarizerOutput, LeafSummarizerShape, LocalTrunkShape,
        ReadFusionShape, TreeMergeCellShape,
    };

    type TestBackend = Candle<f32, i64>;

    fn test_model_shape() -> FractalV2ModelShape {
        test_model_shape_with_leaf_size(16)
    }

    fn test_model_shape_with_leaf_size(leaf_size: usize) -> FractalV2ModelShape {
        FractalV2ModelShape {
            vocab_size: 32_000,
            token_dim: 128,
            local_trunk: LocalTrunkShape {
                token_dim: 128,
                root_count: 2,
                root_state_dim: 96,
                root_readout_dim: 64,
                leaf_size,
            },
            leaf_summarizer: LeafSummarizerShape {
                readout_dim: 64,
                leaf_size,
                summary_dim: 80,
                key_dim: 48,
                value_dim: 72,
                token_cache_key_dim: 40,
                token_cache_value_dim: 56,
            },
            tree_merge_cell: TreeMergeCellShape {
                summary_dim: 80,
                key_dim: 48,
                value_dim: 72,
                scale_embedding_dim: 12,
            },
            router: FractalRouterHeadShape {
                query_dim: 64,
                key_dim: 48,
                head_count: 4,
                beam_width: 2,
                top_leaf_reads: 2,
                allow_early_stop: false,
            },
            exact_read: ExactLeafReadShape {
                query_dim: 64,
                key_dim: 40,
                value_dim: 56,
                head_count: 4,
                top_leaf_reads: 2,
                leaf_size,
            },
            read_fusion: ReadFusionShape {
                root_count: 2,
                root_readout_dim: 64,
                routed_value_dim: 72,
                exact_read_value_dim: 56,
                fused_readout_dim: 96,
            },
        }
    }

    fn test_leaf_summarizer(
        device: &<TestBackend as Backend>::Device,
    ) -> BaselineLeafSummarizer<TestBackend> {
        test_leaf_summarizer_with_leaf_size(16, device)
    }

    fn test_leaf_summarizer_with_leaf_size(
        leaf_size: usize,
        device: &<TestBackend as Backend>::Device,
    ) -> BaselineLeafSummarizer<TestBackend> {
        BaselineLeafSummarizerConfig::new(64, leaf_size, 80, 48, 72, 40, 56).init(device)
    }

    fn test_tree_merge_cell(
        device: &<TestBackend as Backend>::Device,
    ) -> BaselineTreeMergeCell<TestBackend> {
        BaselineTreeMergeCellConfig::new(80, 48, 72, 12).init(device)
    }

    #[derive(Module, Debug, Clone, Copy)]
    struct ZeroLeafSummarizer {
        readout_dim: usize,
        leaf_size: usize,
        summary_dim: usize,
        key_dim: usize,
        value_dim: usize,
        token_cache_key_dim: usize,
        token_cache_value_dim: usize,
    }

    impl ZeroLeafSummarizer {
        fn new(shape: LeafSummarizerShape) -> Self {
            Self {
                readout_dim: shape.readout_dim,
                leaf_size: shape.leaf_size,
                summary_dim: shape.summary_dim,
                key_dim: shape.key_dim,
                value_dim: shape.value_dim,
                token_cache_key_dim: shape.token_cache_key_dim,
                token_cache_value_dim: shape.token_cache_value_dim,
            }
        }
    }

    impl LeafSummarizer<TestBackend> for ZeroLeafSummarizer {
        fn shape(&self) -> LeafSummarizerShape {
            LeafSummarizerShape {
                readout_dim: self.readout_dim,
                leaf_size: self.leaf_size,
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                token_cache_key_dim: self.token_cache_key_dim,
                token_cache_value_dim: self.token_cache_value_dim,
            }
        }

        fn summarize_sealed_leaf(
            &self,
            token_readouts: Tensor<TestBackend, 4>,
        ) -> Result<LeafSummarizerOutput<TestBackend>, FractalError> {
            let [batch_size, _root_count, leaf_size, readout_dim] = token_readouts.dims();
            ensure_match("zero_leaf_summarizer.leaf_size", leaf_size, self.leaf_size)?;
            ensure_match(
                "zero_leaf_summarizer.readout_dim",
                readout_dim,
                self.readout_dim,
            )?;

            Ok(LeafSummarizerOutput::new(
                Tensor::<TestBackend, 2>::zeros(
                    [batch_size, self.summary_dim],
                    &token_readouts.device(),
                ),
                Tensor::<TestBackend, 2>::zeros(
                    [batch_size, self.key_dim],
                    &token_readouts.device(),
                ),
                Tensor::<TestBackend, 2>::zeros(
                    [batch_size, self.value_dim],
                    &token_readouts.device(),
                ),
                Tensor::<TestBackend, 3>::zeros(
                    [batch_size, leaf_size, self.token_cache_key_dim],
                    &token_readouts.device(),
                ),
                Tensor::<TestBackend, 3>::zeros(
                    [batch_size, leaf_size, self.token_cache_value_dim],
                    &token_readouts.device(),
                ),
                Tensor::<TestBackend, 2, Bool>::ones(
                    [batch_size, leaf_size],
                    &token_readouts.device(),
                ),
            ))
        }
    }

    #[derive(Module, Debug, Clone, Copy)]
    struct ArithmeticTreeMergeCell {
        summary_dim: usize,
        key_dim: usize,
        value_dim: usize,
        scale_embedding_dim: usize,
        _marker: PhantomData<TestBackend>,
    }

    impl ArithmeticTreeMergeCell {
        fn new(shape: TreeMergeCellShape) -> Self {
            Self {
                summary_dim: shape.summary_dim,
                key_dim: shape.key_dim,
                value_dim: shape.value_dim,
                scale_embedding_dim: shape.scale_embedding_dim,
                _marker: PhantomData,
            }
        }
    }

    impl TreeMergeCell<TestBackend> for ArithmeticTreeMergeCell {
        fn shape(&self) -> TreeMergeCellShape {
            TreeMergeCellShape {
                summary_dim: self.summary_dim,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                scale_embedding_dim: self.scale_embedding_dim,
            }
        }

        fn merge_pair(
            &self,
            left: TreeNodeBatch<TestBackend>,
            right: TreeNodeBatch<TestBackend>,
            level: usize,
        ) -> Result<crate::v2::TreeMergeOutput<TestBackend>, FractalError> {
            let [batch_size, summary_dim] = left.summary().dims();
            let [right_batch_size, right_summary_dim] = right.summary().dims();
            ensure_match(
                "arithmetic_tree_merge_cell.right_batch_size",
                right_batch_size,
                batch_size,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.right_summary_dim",
                right_summary_dim,
                summary_dim,
            )?;
            let [left_key_batch_size, left_key_dim] = left.key().dims();
            let [right_key_batch_size, right_key_dim] = right.key().dims();
            let [left_value_batch_size, left_value_dim] = left.value().dims();
            let [right_value_batch_size, right_value_dim] = right.value().dims();
            ensure_match(
                "arithmetic_tree_merge_cell.left_key_batch_size",
                left_key_batch_size,
                batch_size,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.right_key_batch_size",
                right_key_batch_size,
                batch_size,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.left_value_batch_size",
                left_value_batch_size,
                batch_size,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.right_value_batch_size",
                right_value_batch_size,
                batch_size,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.left_key_dim",
                left_key_dim,
                self.key_dim,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.right_key_dim",
                right_key_dim,
                self.key_dim,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.left_value_dim",
                left_value_dim,
                self.value_dim,
            )?;
            ensure_match(
                "arithmetic_tree_merge_cell.right_value_dim",
                right_value_dim,
                self.value_dim,
            )?;

            let level_bias = level as f64 + 1.0;
            Ok(crate::v2::TreeMergeOutput::new(
                (left.summary() + right.summary()).add_scalar(level_bias),
                (left.key() + right.key()).add_scalar(level_bias),
                (left.value() + right.value()).add_scalar(level_bias),
            ))
        }
    }

    fn root_readouts_for_token(
        token_index: usize,
        device: &<TestBackend as Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        let base = token_index as f32 + 1.0;
        let mut values = Vec::with_capacity(2 * 2 * 64);
        for batch_index in 0..2 {
            for root_index in 0..2 {
                for readout_index in 0..64 {
                    values.push(
                        base + batch_index as f32 * 0.1
                            + root_index as f32 * 0.01
                            + readout_index as f32 * 0.001,
                    );
                }
            }
        }
        Tensor::<TestBackend, 3>::from_data(TensorData::new(values, [2, 2, 64]), device)
    }

    fn reference_tree_from_leaf_store(
        store: &LeafSummaryStore<TestBackend>,
        cell: &ArithmeticTreeMergeCell,
    ) -> Vec<(Vec<TokenSpan>, Vec<TreeNodeBatch<TestBackend>>)> {
        let [batch_size, leaf_count, summary_dim] = store.summaries().dims();
        let [_, _, key_dim] = store.keys().dims();
        let [_, _, value_dim] = store.values().dims();
        let mut current_spans = store.shared_spans().to_vec();
        let mut current_nodes = Vec::with_capacity(leaf_count);
        for index in 0..leaf_count {
            current_nodes.push(
                TreeNodeBatch::from_tensors(
                    store
                        .summaries()
                        .narrow(1, index, 1)
                        .reshape([batch_size, summary_dim]),
                    store
                        .keys()
                        .narrow(1, index, 1)
                        .reshape([batch_size, key_dim]),
                    store
                        .values()
                        .narrow(1, index, 1)
                        .reshape([batch_size, value_dim]),
                )
                .unwrap(),
            );
        }

        let mut levels = vec![(current_spans.clone(), current_nodes.clone())];
        let mut level_index = 1usize;
        while current_nodes.len() > 1 {
            let mut next_spans = Vec::with_capacity(current_nodes.len().div_ceil(2));
            let mut next_nodes = Vec::with_capacity(current_nodes.len().div_ceil(2));
            for pair_start in (0..current_nodes.len()).step_by(2) {
                if pair_start + 1 < current_nodes.len() {
                    let merged = cell
                        .merge_pair(
                            current_nodes[pair_start].clone(),
                            current_nodes[pair_start + 1].clone(),
                            level_index,
                        )
                        .unwrap()
                        .into_node()
                        .unwrap();
                    next_spans.push(
                        TokenSpan::new(
                            current_spans[pair_start].start(),
                            current_spans[pair_start + 1].end(),
                        )
                        .unwrap(),
                    );
                    next_nodes.push(merged);
                } else {
                    next_spans.push(current_spans[pair_start]);
                    next_nodes.push(current_nodes[pair_start].clone());
                }
            }
            levels.push((next_spans.clone(), next_nodes.clone()));
            current_spans = next_spans;
            current_nodes = next_nodes;
            level_index += 1;
        }

        levels
    }

    fn assert_tree_matches_reference(
        tree: &TreeSummaryState<TestBackend>,
        reference: &[(Vec<TokenSpan>, Vec<TreeNodeBatch<TestBackend>>)],
    ) {
        assert_eq!(tree.levels().len(), reference.len());
        for (level_index, (expected_spans, expected_nodes)) in reference.iter().enumerate() {
            let level = &tree.levels()[level_index];
            assert_eq!(level.shared_spans(), expected_spans.as_slice());
            assert_eq!(level.node_count(), expected_nodes.len());

            for (node_index, expected_node) in expected_nodes.iter().enumerate() {
                let actual_node = level.node(node_index).unwrap();
                assert_eq!(
                    actual_node.summary().to_data().convert::<f32>(),
                    expected_node.summary().to_data().convert::<f32>()
                );
                assert_eq!(
                    actual_node.key().to_data().convert::<f32>(),
                    expected_node.key().to_data().convert::<f32>()
                );
                assert_eq!(
                    actual_node.value().to_data().convert::<f32>(),
                    expected_node.value().to_data().convert::<f32>()
                );
            }
        }
    }

    #[test]
    fn token_span_rejects_inverted_bounds() {
        let error = TokenSpan::new(5, 4).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("token span end"))
        );
    }

    #[test]
    fn retrieval_policy_comes_from_router_shape() {
        let policy = RetrievalPolicy::from_router_shape(test_model_shape().router).unwrap();

        assert_eq!(policy.beam_width(), 2);
        assert_eq!(policy.top_k_reads(), 2);
        assert!(!policy.allow_early_stop());
    }

    #[test]
    fn multi_root_state_from_tensors_rejects_zero_batch_size() {
        let device = <TestBackend as Backend>::Device::default();
        let error = MultiRootState::from_tensors(
            Tensor::<TestBackend, 3>::zeros([0, 2, 4], &device),
            Tensor::<TestBackend, 3>::zeros([0, 2, 3], &device),
            Tensor::<TestBackend, 3>::zeros([0, 2, 3], &device),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("multi_root.batch_size"))
        );
    }

    #[test]
    fn multi_root_state_from_tensors_rejects_zero_root_count() {
        let device = <TestBackend as Backend>::Device::default();
        let error = MultiRootState::from_tensors(
            Tensor::<TestBackend, 3>::zeros([1, 0, 4], &device),
            Tensor::<TestBackend, 3>::zeros([1, 0, 3], &device),
            Tensor::<TestBackend, 3>::zeros([1, 0, 3], &device),
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("multi_root.root_count"))
        );
    }

    #[test]
    fn state_layout_preserves_model_contract_dimensions() {
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 3).unwrap();

        assert_eq!(
            layout,
            FractalV2StateLayout {
                batch_size: 3,
                batch_timeline_mode: BatchTimelineMode::LockstepSharedTimeline,
                root_count: 2,
                root_state_dim: 96,
                root_readout_dim: 64,
                leaf_size: 16,
                summary_dim: 80,
                key_dim: 48,
                value_dim: 72,
                token_cache_key_dim: 40,
                token_cache_value_dim: 56,
                scale_embedding_dim: 12,
            }
        );
    }

    #[test]
    fn fractal_v2_state_initializes_empty_stores_with_explicit_shapes() {
        let device = <TestBackend as Backend>::Device::default();
        let shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            shape,
            3,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = FractalV2StateLayout::from_model_shape(shape, 3).unwrap();

        assert_eq!(
            state.shape(),
            FractalV2StateShape {
                layout,
                roots: MultiRootStateShape {
                    batch_size: 3,
                    root_count: 2,
                    recurrent_dim: 96,
                    intent_dim: 64,
                },
                live_leaf: LiveLeafStateShape {
                    batch_size: 3,
                    root_count: 2,
                    tokens_per_leaf: 16,
                    readout_dim: 64,
                    batch_timeline_mode: BatchTimelineMode::LockstepSharedTimeline,
                    shared_span: TokenSpan::empty_at(0),
                    shared_valid_tokens: 0,
                },
                sealed_leaves: LeafSummaryStoreShape {
                    batch_size: 3,
                    leaf_count: 0,
                    summary_dim: 80,
                    key_dim: 48,
                    value_dim: 72,
                    batch_timeline_mode: BatchTimelineMode::LockstepSharedTimeline,
                },
                tree: TreeSummaryStateShape { levels: Vec::new() },
                leaf_token_cache: LeafTokenCacheShape {
                    batch_size: 3,
                    leaf_count: 0,
                    tokens_per_leaf: 16,
                    key_dim: 40,
                    value_dim: 56,
                    batch_timeline_mode: BatchTimelineMode::LockstepSharedTimeline,
                },
                retrieval_policy: RetrievalPolicy::from_router_shape(shape.router).unwrap(),
                merge_policy: MergeCheckpointPolicy::FixedLeafSize {
                    tokens_per_leaf: 16
                },
            }
        );
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_updates_live_leaf_without_future_leakage() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);

        let sealed = state
            .append_root_readouts(
                root_readouts_for_token(0, &device),
                &summarizer,
                &tree_merge_cell,
            )
            .unwrap();

        assert!(sealed.is_none());
        assert_eq!(
            state.live_leaf().shared_span(),
            TokenSpan::new(0, 1).unwrap()
        );
        assert_eq!(state.live_leaf().shared_valid_tokens(), 1);
        assert!(state.sealed_leaves().shared_spans().is_empty());
        assert!(state.leaf_token_cache().shared_spans().is_empty());
        assert!(state.tree().levels().is_empty());

        let live_data = state
            .live_leaf()
            .token_readouts()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        let readout_stride = 64;
        let token_stride = 16 * readout_stride;
        let root_stride = token_stride;
        let batch_stride = 2 * root_stride;
        for batch_index in 0..2 {
            for root_index in 0..2 {
                for token_index in 1..16 {
                    for readout_index in 0..64 {
                        let flat_index = batch_index * batch_stride
                            + root_index * root_stride
                            + token_index * readout_stride
                            + readout_index;
                        assert_eq!(live_data[flat_index], 0.0);
                    }
                }
            }
        }
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_seals_leaf_and_populates_cache() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);
        let mut sealed_leaf = None;

        for token_index in 0..16 {
            sealed_leaf = state
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        let sealed_leaf = sealed_leaf.expect("the 16th append should seal the leaf");
        assert_eq!(sealed_leaf.leaf_index(), 0);
        assert_eq!(sealed_leaf.shared_span(), TokenSpan::new(0, 16).unwrap());
        assert_eq!(sealed_leaf.summary().dims(), [2, 80]);
        assert_eq!(sealed_leaf.key().dims(), [2, 48]);
        assert_eq!(sealed_leaf.value().dims(), [2, 72]);
        assert_eq!(sealed_leaf.token_keys().dims(), [2, 16, 40]);
        assert_eq!(sealed_leaf.token_values().dims(), [2, 16, 56]);
        assert_eq!(sealed_leaf.token_mask().dims(), [2, 16]);

        assert_eq!(state.live_leaf().shared_span(), TokenSpan::empty_at(16));
        assert_eq!(state.live_leaf().shared_valid_tokens(), 0);
        assert_eq!(
            state.sealed_leaves().shared_spans(),
            &[TokenSpan::new(0, 16).unwrap()]
        );
        assert_eq!(
            state.leaf_token_cache().shared_spans(),
            &[TokenSpan::new(0, 16).unwrap()]
        );
        assert_eq!(state.tree().levels().len(), 1);
        assert_eq!(
            state.tree().levels()[0].shared_spans(),
            &[TokenSpan::new(0, 16).unwrap()]
        );
        assert_eq!(state.sealed_leaves().summaries().dims(), [2, 1, 80]);
        assert_eq!(state.leaf_token_cache().keys().dims(), [2, 1, 16, 40]);
        assert_eq!(state.leaf_token_cache().values().dims(), [2, 1, 16, 56]);
        assert_eq!(count_true(state.leaf_token_cache().mask()), 32);
        assert_eq!(
            state.leaf_token_cache().keys().to_data().convert::<f32>(),
            sealed_leaf
                .token_keys()
                .reshape([2, 1, 16, 40])
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            state.leaf_token_cache().values().to_data().convert::<f32>(),
            sealed_leaf
                .token_values()
                .reshape([2, 1, 16, 56])
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            state.leaf_token_cache().mask().to_data().convert::<bool>(),
            sealed_leaf
                .token_mask()
                .reshape([2, 1, 16])
                .to_data()
                .convert::<bool>()
        );
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_with_active_root_count_materializes_ablated_memory() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);
        let mut expected_tokens = Vec::with_capacity(16);
        let mut sealed_leaf = None;

        for token_index in 0..16 {
            let root_readouts = root_readouts_for_token(token_index, &device);
            expected_tokens.push(root_readouts.clone().narrow(1, 0, 1).reshape([2, 1, 1, 64]));
            sealed_leaf = state
                .append_root_readouts_with_active_root_count(
                    root_readouts,
                    1,
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        let sealed_leaf = sealed_leaf.expect("the 16th append should seal the leaf");
        let expected = summarizer
            .summarize_sealed_leaf(Tensor::cat(expected_tokens, 2))
            .unwrap();

        assert_eq!(
            sealed_leaf.summary().to_data().convert::<f32>(),
            expected.summary().to_data().convert::<f32>()
        );
        assert_eq!(
            sealed_leaf.key().to_data().convert::<f32>(),
            expected.key().to_data().convert::<f32>()
        );
        assert_eq!(
            sealed_leaf.value().to_data().convert::<f32>(),
            expected.value().to_data().convert::<f32>()
        );
        assert_eq!(
            sealed_leaf.token_keys().to_data().convert::<f32>(),
            expected.token_keys().to_data().convert::<f32>()
        );
        assert_eq!(
            sealed_leaf.token_values().to_data().convert::<f32>(),
            expected.token_values().to_data().convert::<f32>()
        );
        assert_eq!(
            sealed_leaf.token_mask().to_data().convert::<bool>(),
            expected.token_mask().to_data().convert::<bool>()
        );
        assert_eq!(
            state.sealed_leaves().summaries().to_data().convert::<f32>(),
            expected
                .summary()
                .reshape([2, 1, 80])
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            state.sealed_leaves().values().to_data().convert::<f32>(),
            expected
                .value()
                .reshape([2, 1, 72])
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            state.tree().levels()[0].values().to_data().convert::<f32>(),
            expected
                .value()
                .reshape([2, 1, 72])
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            state.leaf_token_cache().values().to_data().convert::<f32>(),
            expected
                .token_values()
                .reshape([2, 1, 16, 56])
                .to_data()
                .convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_starts_new_live_leaf_after_seal() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);

        for token_index in 0..16 {
            state
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        let seventeenth = root_readouts_for_token(16, &device);
        let sealed = state
            .append_root_readouts(seventeenth.clone(), &summarizer, &tree_merge_cell)
            .unwrap();

        assert!(sealed.is_none());
        assert_eq!(
            state.live_leaf().shared_span(),
            TokenSpan::new(16, 17).unwrap()
        );
        assert_eq!(state.live_leaf().shared_valid_tokens(), 1);
        assert_eq!(
            state
                .live_leaf()
                .token_readouts()
                .narrow(2, 0, 1)
                .reshape([2, 2, 64])
                .to_data()
                .convert::<f32>(),
            seventeenth.to_data().convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_state_leaf_sealing_is_deterministic() {
        let device = <TestBackend as Backend>::Device::default();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);
        let build_state = || {
            FractalV2State::<TestBackend>::for_model_shape(
                test_model_shape(),
                2,
                MergeCheckpointPolicy::FixedLeafSize {
                    tokens_per_leaf: 16,
                },
                &device,
            )
            .unwrap()
        };
        let mut first = build_state();
        let mut second = build_state();

        for token_index in 0..16 {
            first
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
            second
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        assert_eq!(
            first.sealed_leaves().summaries().to_data().convert::<f32>(),
            second
                .sealed_leaves()
                .summaries()
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(
            first.sealed_leaves().keys().to_data().convert::<f32>(),
            second.sealed_leaves().keys().to_data().convert::<f32>()
        );
        assert_eq!(
            first.sealed_leaves().values().to_data().convert::<f32>(),
            second.sealed_leaves().values().to_data().convert::<f32>()
        );
        assert_eq!(
            first.leaf_token_cache().keys().to_data().convert::<f32>(),
            second.leaf_token_cache().keys().to_data().convert::<f32>()
        );
        assert_eq!(
            first.leaf_token_cache().values().to_data().convert::<f32>(),
            second
                .leaf_token_cache()
                .values()
                .to_data()
                .convert::<f32>()
        );
        assert_eq!(first.tree().shape(), second.tree().shape());
        assert_eq!(
            first.tree().levels()[0]
                .summaries()
                .to_data()
                .convert::<f32>(),
            second.tree().levels()[0]
                .summaries()
                .to_data()
                .convert::<f32>()
        );
    }

    #[test]
    fn fractal_v2_state_tree_matches_reference_recompute_after_incremental_seals() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = ArithmeticTreeMergeCell::new(test_model_shape().tree_merge_cell);

        for token_index in 0..64 {
            state
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        let reference = reference_tree_from_leaf_store(state.sealed_leaves(), &tree_merge_cell);

        assert_tree_matches_reference(state.tree(), &reference);
        assert_eq!(state.tree().diagnostics().nodes_per_level, vec![4, 2, 1]);
        assert_eq!(state.tree().diagnostics().tree_depth_reached, 3);
        assert!(!state.tree().diagnostics().has_dead_or_unused_nodes);
    }

    #[test]
    fn fractal_v2_state_tree_matches_reference_recompute_across_leaf_sizes_after_frontier_carry() {
        let device = <TestBackend as Backend>::Device::default();

        for leaf_size in [16usize, 32, 64] {
            let model_shape = test_model_shape_with_leaf_size(leaf_size);
            let mut state = FractalV2State::<TestBackend>::for_model_shape(
                model_shape,
                2,
                MergeCheckpointPolicy::FixedLeafSize {
                    tokens_per_leaf: leaf_size,
                },
                &device,
            )
            .unwrap();
            let summarizer = ZeroLeafSummarizer::new(model_shape.leaf_summarizer);
            let tree_merge_cell = ArithmeticTreeMergeCell::new(model_shape.tree_merge_cell);

            for token_index in 0..(leaf_size * 8) {
                let sealed_leaf = state
                    .append_root_readouts(
                        root_readouts_for_token(token_index, &device),
                        &summarizer,
                        &tree_merge_cell,
                    )
                    .unwrap();

                if sealed_leaf.is_some() {
                    let reference =
                        reference_tree_from_leaf_store(state.sealed_leaves(), &tree_merge_cell);
                    assert_tree_matches_reference(state.tree(), &reference);
                    assert!(
                        !state.tree().diagnostics().has_dead_or_unused_nodes,
                        "leaf_size={leaf_size} sealed_leaf_count={}",
                        state.sealed_leaves().shared_spans().len()
                    );
                }
            }

            assert_eq!(state.sealed_leaves().shared_spans().len(), 8);
        }
    }

    #[test]
    fn fractal_v2_state_tree_carries_odd_frontier_spans_without_future_leakage() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = ArithmeticTreeMergeCell::new(test_model_shape().tree_merge_cell);

        for token_index in 0..48 {
            state
                .append_root_readouts(
                    root_readouts_for_token(token_index, &device),
                    &summarizer,
                    &tree_merge_cell,
                )
                .unwrap();
        }

        let level_spans: Vec<Vec<TokenSpan>> = state
            .tree()
            .levels()
            .iter()
            .map(|level| level.shared_spans().to_vec())
            .collect();
        assert_eq!(
            level_spans,
            vec![
                vec![
                    TokenSpan::new(0, 16).unwrap(),
                    TokenSpan::new(16, 32).unwrap(),
                    TokenSpan::new(32, 48).unwrap()
                ],
                vec![
                    TokenSpan::new(0, 32).unwrap(),
                    TokenSpan::new(32, 48).unwrap()
                ],
                vec![TokenSpan::new(0, 48).unwrap()],
            ]
        );
        assert_eq!(state.live_leaf().shared_span(), TokenSpan::empty_at(48));
        assert_eq!(state.live_leaf().shared_valid_tokens(), 0);
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_rejects_mismatched_readout_dim() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let tree_merge_cell = test_tree_merge_cell(&device);
        let invalid = Tensor::<TestBackend, 3>::zeros([2, 2, 63], &device);

        let error = state
            .append_root_readouts(invalid, &summarizer, &tree_merge_cell)
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("state.append_root_readouts.readout_dim"))
        );
    }

    #[test]
    fn fractal_v2_state_append_root_readouts_rejects_mismatched_tree_merge_shape() {
        let device = <TestBackend as Backend>::Device::default();
        let mut state = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let summarizer = test_leaf_summarizer(&device);
        let invalid_tree_merge_cell = ArithmeticTreeMergeCell::new(TreeMergeCellShape {
            summary_dim: 79,
            key_dim: 48,
            value_dim: 72,
            scale_embedding_dim: 12,
        });

        let error = state
            .append_root_readouts(
                root_readouts_for_token(0, &device),
                &summarizer,
                &invalid_tree_merge_cell,
            )
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("state.append_root_readouts.tree_merge_cell.summary_dim"))
        );
    }

    #[test]
    fn leaf_summary_store_push_rejects_non_fixed_sealed_span_width() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let mut store = LeafSummaryStore::<TestBackend>::empty(layout, &device);

        let error = store
            .push_sealed_leaf(
                Tensor::<TestBackend, 2>::zeros([layout.batch_size, layout.summary_dim], &device),
                Tensor::<TestBackend, 2>::zeros([layout.batch_size, layout.key_dim], &device),
                Tensor::<TestBackend, 2>::zeros([layout.batch_size, layout.value_dim], &device),
                layout.leaf_size,
                TokenSpan::new(0, layout.leaf_size / 2).unwrap(),
            )
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("leaf_summary.push.shared_span.len"))
        );
    }

    #[test]
    fn leaf_token_cache_push_rejects_non_fixed_sealed_span_width() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let mut cache = LeafTokenCache::<TestBackend>::empty(layout, &device);

        let error = cache
            .push_sealed_leaf(
                Tensor::<TestBackend, 3>::zeros(
                    [
                        layout.batch_size,
                        layout.leaf_size,
                        layout.token_cache_key_dim,
                    ],
                    &device,
                ),
                Tensor::<TestBackend, 3>::zeros(
                    [
                        layout.batch_size,
                        layout.leaf_size,
                        layout.token_cache_value_dim,
                    ],
                    &device,
                ),
                Tensor::<TestBackend, 2, Int>::ones([layout.batch_size, layout.leaf_size], &device)
                    .greater_elem(0),
                layout.leaf_size,
                TokenSpan::new(0, layout.leaf_size / 2).unwrap(),
            )
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("leaf_token_cache.push.shared_span.len"))
        );
    }

    #[test]
    fn fractal_v2_state_rejects_merge_policy_leaf_size_mismatch() {
        let device = <TestBackend as Backend>::Device::default();
        let error = FractalV2State::<TestBackend>::for_model_shape(
            test_model_shape(),
            3,
            MergeCheckpointPolicy::FixedLeafSize { tokens_per_leaf: 8 },
            &device,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("merge_checkpoint_policy.tokens_per_leaf"))
        );
    }

    #[test]
    fn tree_summary_state_enforces_contiguous_level_insertion_and_layout() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let mut tree = TreeSummaryState::<TestBackend>::empty(layout);
        let error = tree
            .upsert_level(TreeLevelStore::empty(1, layout, &device))
            .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidState(message) if message.contains("cannot insert tree level 1"))
        );

        let mismatched_level = TreeLevelStore::from_record(
            TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size + 1, 0, layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size + 1, 0, layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size + 1, 0, layout.value_dim],
                    &device,
                ),
                level: 0,
                shared_spans: Vec::new(),
            },
            layout,
        )
        .unwrap_err();

        assert!(
            matches!(mismatched_level, FractalError::InvalidConfig(message) if message.contains("tree_level.summaries"))
        );
    }

    #[test]
    fn tree_summary_state_record_requires_complete_dyadic_chain() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let spans = vec![
            TokenSpan::new(0, layout.leaf_size).unwrap(),
            TokenSpan::new(layout.leaf_size, layout.leaf_size * 2).unwrap(),
        ];

        let error = TreeSummaryState::<TestBackend>::from_record(
            TreeSummaryStateRecord {
                levels: vec![TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, spans.len(), layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, spans.len(), layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, spans.len(), layout.value_dim],
                        &device,
                    ),
                    level: 0,
                    shared_spans: spans,
                }],
            },
            layout,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("missing one or more deterministic parent levels"))
        );
    }

    #[test]
    fn tree_summary_state_upsert_rolls_back_inconsistent_level_replacement() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let mut tree = TreeSummaryState::<TestBackend>::empty(layout);
        let level0 = TreeLevelStore::from_record(
            TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 2, layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 2, layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 2, layout.value_dim],
                    &device,
                ),
                level: 0,
                shared_spans: vec![
                    TokenSpan::new(0, layout.leaf_size).unwrap(),
                    TokenSpan::new(layout.leaf_size, layout.leaf_size * 2).unwrap(),
                ],
            },
            layout,
        )
        .unwrap();
        let level1 = TreeLevelStore::from_record(
            TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.value_dim],
                    &device,
                ),
                level: 1,
                shared_spans: vec![TokenSpan::new(0, layout.leaf_size * 2).unwrap()],
            },
            layout,
        )
        .unwrap();
        tree.upsert_level(level0).unwrap();
        tree.upsert_level(level1).unwrap();

        let replacement = TreeLevelStore::from_record(
            TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 3, layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 3, layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 3, layout.value_dim],
                    &device,
                ),
                level: 0,
                shared_spans: vec![
                    TokenSpan::new(0, layout.leaf_size).unwrap(),
                    TokenSpan::new(layout.leaf_size, layout.leaf_size * 2).unwrap(),
                    TokenSpan::new(layout.leaf_size * 2, layout.leaf_size * 3).unwrap(),
                ],
            },
            layout,
        )
        .unwrap();

        let error = tree.upsert_level(replacement).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("tree level 1 spans do not match"))
        );
        assert_eq!(tree.levels()[0].shared_spans().len(), 2);
        assert_eq!(tree.levels()[1].shared_spans().len(), 1);
    }

    #[test]
    fn fractal_v2_state_record_roundtrips() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let original = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();

        let restored =
            FractalV2State::<TestBackend>::from_record(original.clone().into_record(), model_shape)
                .unwrap();

        assert_eq!(restored.shape(), original.shape());
    }

    #[test]
    fn fractal_v2_state_record_supports_burn_record_roundtrip() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let original = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();

        let item = Record::<TestBackend>::into_item::<FullPrecisionSettings>(
            original.clone().into_record(),
        );
        let restored_record =
            FractalV2StateRecord::<TestBackend>::from_item::<FullPrecisionSettings>(item, &device);
        let restored =
            FractalV2State::<TestBackend>::from_record(restored_record, model_shape).unwrap();

        assert_eq!(restored.shape(), original.shape());
    }

    #[test]
    fn fractal_v2_state_record_rejects_leaf_cache_drift() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let mut record = state.into_record();
        record
            .leaf_token_cache
            .shared_spans
            .push(TokenSpan::new(0, 16).unwrap());

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(ref message) if message.contains("leaf_token_cache.shared_spans[0]"))
                || matches!(error, FractalError::InvalidConfig(ref message) if message.contains("leaf_token_cache.shared_span_count"))
                || matches!(error, FractalError::InvalidConfig(ref message) if message.contains("sealed leaf summaries and leaf token cache"))
        );
    }

    #[test]
    fn fractal_v2_state_record_rejects_retrieval_policy_drift() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let mut record = state.into_record();
        record.retrieval_policy = RetrievalPolicy::from_router_shape(FractalRouterHeadShape {
            query_dim: model_shape.router.query_dim,
            key_dim: model_shape.router.key_dim,
            head_count: model_shape.router.head_count,
            beam_width: model_shape.router.beam_width + 1,
            top_leaf_reads: model_shape.router.top_leaf_reads,
            allow_early_stop: false,
        })
        .unwrap();

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("state_record.retrieval_policy mismatch"))
        );
    }

    #[test]
    fn fractal_v2_state_rejects_model_shape_that_breaks_cross_component_contracts() {
        let device = <TestBackend as Backend>::Device::default();
        let mut invalid_shape = test_model_shape();
        invalid_shape.router.query_dim += 1;

        let error = FractalV2State::<TestBackend>::for_model_shape(
            invalid_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("router.query_dim"))
        );
    }

    #[test]
    fn fractal_v2_state_rejects_early_stop_router_shapes_in_v1() {
        let device = <TestBackend as Backend>::Device::default();
        let mut invalid_shape = test_model_shape();
        invalid_shape.router.allow_early_stop = true;

        let error = FractalV2State::<TestBackend>::for_model_shape(
            invalid_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("router.allow_early_stop"))
        );
    }

    #[test]
    fn fractal_v2_state_record_rejects_live_leaf_that_overlaps_sealed_frontier() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = state.layout();
        let mut record = state.into_record();
        record.sealed_leaves = LeafSummaryStoreRecord {
            summaries: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.summary_dim],
                &device,
            ),
            keys: Tensor::<TestBackend, 3>::zeros([layout.batch_size, 1, layout.key_dim], &device),
            values: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.value_dim],
                &device,
            ),
            shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
        };
        record.leaf_token_cache = LeafTokenCacheRecord {
            keys: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                &device,
            ),
            values: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                &device,
            ),
            mask: Tensor::<TestBackend, 3, Int>::ones(
                [layout.batch_size, 1, layout.leaf_size],
                &device,
            )
            .greater_elem(0),
            shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
        };
        record.live_leaf = LiveLeafStateRecord {
            token_readouts: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                &device,
            ),
            shared_span: TokenSpan::new(0, 8).unwrap(),
            shared_valid_tokens: 8,
        };
        record.tree = TreeSummaryStateRecord {
            levels: vec![TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, 1, layout.value_dim],
                    &device,
                ),
                level: 0,
                shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
            }],
        };

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("live_leaf.shared_span.start"))
        );
    }

    #[test]
    fn live_leaf_state_record_rejects_nonzero_tail_past_shared_valid_tokens() {
        use burn::tensor::TensorData;

        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let mut values =
            vec![
                0.0f32;
                layout.batch_size * layout.root_count * layout.leaf_size * layout.root_readout_dim
            ];
        let readout_stride = layout.root_readout_dim;
        let token_stride = layout.leaf_size * readout_stride;
        let root_stride = layout.root_count * token_stride;
        let tail_index = root_stride + (2 * readout_stride);
        values[tail_index] = 1.0;

        let error = LiveLeafState::<TestBackend>::from_record(
            LiveLeafStateRecord {
                token_readouts: Tensor::<TestBackend, 4>::from_data(
                    TensorData::new(
                        values,
                        [
                            layout.batch_size,
                            layout.root_count,
                            layout.leaf_size,
                            layout.root_readout_dim,
                        ],
                    ),
                    &device,
                ),
                shared_span: TokenSpan::new(0, 2).unwrap(),
                shared_valid_tokens: 2,
            },
            layout,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("must be zero beyond shared_valid_tokens"))
        );
    }

    #[test]
    fn tree_summary_state_record_rejects_duplicate_levels() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let level0_span = TokenSpan::new(0, layout.leaf_size).unwrap();
        let duplicate_levels = TreeSummaryState::<TestBackend>::from_record(
            TreeSummaryStateRecord {
                levels: vec![
                    TreeLevelStoreRecord {
                        summaries: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.summary_dim],
                            &device,
                        ),
                        keys: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.key_dim],
                            &device,
                        ),
                        values: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.value_dim],
                            &device,
                        ),
                        level: 0,
                        shared_spans: vec![level0_span],
                    },
                    TreeLevelStoreRecord {
                        summaries: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.summary_dim],
                            &device,
                        ),
                        keys: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.key_dim],
                            &device,
                        ),
                        values: Tensor::<TestBackend, 3>::zeros(
                            [layout.batch_size, 1, layout.value_dim],
                            &device,
                        ),
                        level: 0,
                        shared_spans: vec![level0_span],
                    },
                ],
            },
            layout,
        )
        .unwrap_err();

        assert!(
            matches!(duplicate_levels, FractalError::InvalidConfig(message) if message.contains("duplicate level 0"))
        );
    }

    #[test]
    fn checked_usize_product_rejects_multiplication_overflow() {
        let error = checked_usize_product("overflow", &[usize::MAX, 2]).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("overflow while multiplying by 2"))
        );
    }

    #[test]
    fn fractal_v2_state_record_requires_tree_once_sealed_leaves_exist() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = state.layout();
        let mut record = state.into_record();
        record.sealed_leaves = LeafSummaryStoreRecord {
            summaries: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.summary_dim],
                &device,
            ),
            keys: Tensor::<TestBackend, 3>::zeros([layout.batch_size, 1, layout.key_dim], &device),
            values: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.value_dim],
                &device,
            ),
            shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
        };
        record.leaf_token_cache = LeafTokenCacheRecord {
            keys: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                &device,
            ),
            values: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                &device,
            ),
            mask: Tensor::<TestBackend, 3, Int>::ones(
                [layout.batch_size, 1, layout.leaf_size],
                &device,
            )
            .greater_elem(0),
            shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
        };
        record.live_leaf = LiveLeafStateRecord {
            token_readouts: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                &device,
            ),
            shared_span: TokenSpan::empty_at(layout.leaf_size),
            shared_valid_tokens: 0,
        };

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("tree must include level 0 once sealed leaves exist"))
        );
    }

    #[test]
    fn leaf_token_cache_record_rejects_holey_masks() {
        let device = <TestBackend as Backend>::Device::default();
        let layout = FractalV2StateLayout::from_model_shape(test_model_shape(), 2).unwrap();
        let error = LeafTokenCache::<TestBackend>::from_record(
            LeafTokenCacheRecord {
                keys: Tensor::<TestBackend, 4>::zeros(
                    [
                        layout.batch_size,
                        1,
                        layout.leaf_size,
                        layout.token_cache_key_dim,
                    ],
                    &device,
                ),
                values: Tensor::<TestBackend, 4>::zeros(
                    [
                        layout.batch_size,
                        1,
                        layout.leaf_size,
                        layout.token_cache_value_dim,
                    ],
                    &device,
                ),
                mask: false_mask([layout.batch_size, 1, layout.leaf_size], &device),
                shared_spans: vec![TokenSpan::new(0, layout.leaf_size).unwrap()],
            },
            layout,
        )
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("leaf_token_cache.mask_true_count"))
        );
    }

    #[test]
    fn fractal_v2_state_record_rejects_trailing_empty_tree_levels() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = state.layout();
        let mut record = state.into_record();
        let sealed_span = TokenSpan::new(0, layout.leaf_size).unwrap();
        record.sealed_leaves = LeafSummaryStoreRecord {
            summaries: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.summary_dim],
                &device,
            ),
            keys: Tensor::<TestBackend, 3>::zeros([layout.batch_size, 1, layout.key_dim], &device),
            values: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, 1, layout.value_dim],
                &device,
            ),
            shared_spans: vec![sealed_span],
        };
        record.leaf_token_cache = LeafTokenCacheRecord {
            keys: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                &device,
            ),
            values: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    1,
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                &device,
            ),
            mask: Tensor::<TestBackend, 3, Int>::ones(
                [layout.batch_size, 1, layout.leaf_size],
                &device,
            )
            .greater_elem(0),
            shared_spans: vec![sealed_span],
        };
        record.live_leaf = LiveLeafStateRecord {
            token_readouts: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                &device,
            ),
            shared_span: TokenSpan::empty_at(layout.leaf_size),
            shared_valid_tokens: 0,
        };
        record.tree = TreeSummaryStateRecord {
            levels: vec![
                TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 1, layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 1, layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 1, layout.value_dim],
                        &device,
                    ),
                    level: 0,
                    shared_spans: vec![sealed_span],
                },
                TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 0, layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 0, layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, 0, layout.value_dim],
                        &device,
                    ),
                    level: 1,
                    shared_spans: Vec::new(),
                },
            ],
        };

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("trailing level 1"))
        );
    }

    #[test]
    fn fractal_v2_state_record_rejects_missing_parent_tree_levels() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = state.layout();
        let mut record = state.into_record();
        let spans = vec![
            TokenSpan::new(0, layout.leaf_size).unwrap(),
            TokenSpan::new(layout.leaf_size, layout.leaf_size * 2).unwrap(),
        ];
        record.sealed_leaves = LeafSummaryStoreRecord {
            summaries: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, spans.len(), layout.summary_dim],
                &device,
            ),
            keys: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, spans.len(), layout.key_dim],
                &device,
            ),
            values: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, spans.len(), layout.value_dim],
                &device,
            ),
            shared_spans: spans.clone(),
        };
        record.leaf_token_cache = LeafTokenCacheRecord {
            keys: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    spans.len(),
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                &device,
            ),
            values: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    spans.len(),
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                &device,
            ),
            mask: Tensor::<TestBackend, 3, Int>::ones(
                [layout.batch_size, spans.len(), layout.leaf_size],
                &device,
            )
            .greater_elem(0),
            shared_spans: spans.clone(),
        };
        record.live_leaf = LiveLeafStateRecord {
            token_readouts: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                &device,
            ),
            shared_span: TokenSpan::empty_at(layout.leaf_size * spans.len()),
            shared_valid_tokens: 0,
        };
        record.tree = TreeSummaryStateRecord {
            levels: vec![TreeLevelStoreRecord {
                summaries: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, spans.len(), layout.summary_dim],
                    &device,
                ),
                keys: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, spans.len(), layout.key_dim],
                    &device,
                ),
                values: Tensor::<TestBackend, 3>::zeros(
                    [layout.batch_size, spans.len(), layout.value_dim],
                    &device,
                ),
                level: 0,
                shared_spans: spans,
            }],
        };

        let error = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("missing one or more deterministic parent levels"))
        );
    }

    #[test]
    fn fractal_v2_state_record_accepts_carried_odd_frontier_tree() {
        let device = <TestBackend as Backend>::Device::default();
        let model_shape = test_model_shape();
        let state = FractalV2State::<TestBackend>::for_model_shape(
            model_shape,
            2,
            MergeCheckpointPolicy::FixedLeafSize {
                tokens_per_leaf: 16,
            },
            &device,
        )
        .unwrap();
        let layout = state.layout();
        let mut record = state.into_record();
        let level0_spans = vec![
            TokenSpan::new(0, layout.leaf_size).unwrap(),
            TokenSpan::new(layout.leaf_size, layout.leaf_size * 2).unwrap(),
            TokenSpan::new(layout.leaf_size * 2, layout.leaf_size * 3).unwrap(),
        ];
        let level1_spans = vec![
            TokenSpan::new(0, layout.leaf_size * 2).unwrap(),
            TokenSpan::new(layout.leaf_size * 2, layout.leaf_size * 3).unwrap(),
        ];
        let level2_spans = vec![TokenSpan::new(0, layout.leaf_size * 3).unwrap()];
        record.sealed_leaves = LeafSummaryStoreRecord {
            summaries: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, level0_spans.len(), layout.summary_dim],
                &device,
            ),
            keys: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, level0_spans.len(), layout.key_dim],
                &device,
            ),
            values: Tensor::<TestBackend, 3>::zeros(
                [layout.batch_size, level0_spans.len(), layout.value_dim],
                &device,
            ),
            shared_spans: level0_spans.clone(),
        };
        record.leaf_token_cache = LeafTokenCacheRecord {
            keys: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    level0_spans.len(),
                    layout.leaf_size,
                    layout.token_cache_key_dim,
                ],
                &device,
            ),
            values: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    level0_spans.len(),
                    layout.leaf_size,
                    layout.token_cache_value_dim,
                ],
                &device,
            ),
            mask: Tensor::<TestBackend, 3, Int>::ones(
                [layout.batch_size, level0_spans.len(), layout.leaf_size],
                &device,
            )
            .greater_elem(0),
            shared_spans: level0_spans.clone(),
        };
        record.live_leaf = LiveLeafStateRecord {
            token_readouts: Tensor::<TestBackend, 4>::zeros(
                [
                    layout.batch_size,
                    layout.root_count,
                    layout.leaf_size,
                    layout.root_readout_dim,
                ],
                &device,
            ),
            shared_span: TokenSpan::empty_at(layout.leaf_size * level0_spans.len()),
            shared_valid_tokens: 0,
        };
        record.tree = TreeSummaryStateRecord {
            levels: vec![
                TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level0_spans.len(), layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level0_spans.len(), layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level0_spans.len(), layout.value_dim],
                        &device,
                    ),
                    level: 0,
                    shared_spans: level0_spans,
                },
                TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level1_spans.len(), layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level1_spans.len(), layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level1_spans.len(), layout.value_dim],
                        &device,
                    ),
                    level: 1,
                    shared_spans: level1_spans,
                },
                TreeLevelStoreRecord {
                    summaries: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level2_spans.len(), layout.summary_dim],
                        &device,
                    ),
                    keys: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level2_spans.len(), layout.key_dim],
                        &device,
                    ),
                    values: Tensor::<TestBackend, 3>::zeros(
                        [layout.batch_size, level2_spans.len(), layout.value_dim],
                        &device,
                    ),
                    level: 2,
                    shared_spans: level2_spans,
                },
            ],
        };

        let restored = FractalV2State::<TestBackend>::from_record(record, model_shape).unwrap();

        assert_eq!(restored.shape().tree.levels.len(), 3);
    }
}
