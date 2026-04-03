use burn::prelude::ElementConversion;
use burn::record::{PrecisionSettings, Record};
use burn::tensor::{backend::Backend, Bool, Int, Tensor};
use serde::{Deserialize, Serialize};

use crate::error::FractalError;

use super::{model::FractalV2ModelShape, router::FractalRouterHeadShape};

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
        record::{FullPrecisionSettings, Record},
    };

    use super::*;
    use crate::v2::{
        FractalRouterHeadShape, FractalV2ModelShape, LeafSummarizerShape, LocalTrunkShape,
        ReadFusionShape, TreeMergeCellShape,
    };

    type TestBackend = Candle<f32, i64>;

    fn test_model_shape() -> FractalV2ModelShape {
        FractalV2ModelShape {
            vocab_size: 32_000,
            token_dim: 128,
            local_trunk: LocalTrunkShape {
                token_dim: 128,
                root_count: 2,
                root_state_dim: 96,
                root_readout_dim: 64,
                leaf_size: 16,
            },
            leaf_summarizer: LeafSummarizerShape {
                token_dim: 128,
                leaf_size: 16,
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
            read_fusion: ReadFusionShape {
                root_count: 2,
                root_readout_dim: 64,
                retrieved_value_dim: 56,
                fused_readout_dim: 96,
            },
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
    fn fractal_v2_state_record_rejects_missing_tree_after_sealing_leaf() {
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
