use std::num::NonZeroUsize;

use fractal_core::error::FractalError;

use crate::overlay::{
    pack_overlay_documents_in_batches, OverlayBatchPack, OverlayBatchPackSummary,
    OverlayBatchPackingStrategy, OverlayDictionaryScope, OverlaySharingPolicy,
    PackedOverlayDocumentTransport, RecursiveOverlayDocument,
};

use super::traits::ModelAdapter;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayModelFacingDocument {
    overlay: RecursiveOverlayDocument,
}

impl OverlayModelFacingDocument {
    pub fn new(overlay: RecursiveOverlayDocument) -> Result<Self, FractalError> {
        if !overlay.exact_ok() {
            return Err(FractalError::InvalidState(
                "overlay model-facing document must expand back to its canonical token ids exactly"
                    .to_string(),
            ));
        }
        Ok(Self { overlay })
    }

    pub fn overlay(&self) -> &RecursiveOverlayDocument {
        &self.overlay
    }

    pub fn canonical_token_count(&self) -> usize {
        self.overlay.canonical.token_ids.len()
    }

    pub fn into_overlay(self) -> RecursiveOverlayDocument {
        self.overlay
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OverlayModelFacingBatch {
    documents: Vec<OverlayModelFacingDocument>,
}

impl OverlayModelFacingBatch {
    pub fn new(documents: Vec<OverlayModelFacingDocument>) -> Self {
        Self { documents }
    }

    pub fn singleton(document: OverlayModelFacingDocument) -> Self {
        Self {
            documents: vec![document],
        }
    }

    pub fn documents(&self) -> &[OverlayModelFacingDocument] {
        &self.documents
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }
}

impl From<OverlayModelFacingDocument> for OverlayModelFacingBatch {
    fn from(document: OverlayModelFacingDocument) -> Self {
        Self::singleton(document)
    }
}

impl From<Vec<OverlayModelFacingDocument>> for OverlayModelFacingBatch {
    fn from(documents: Vec<OverlayModelFacingDocument>) -> Self {
        Self::new(documents)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayTransportConfig {
    pub scope: OverlayDictionaryScope,
    pub sharing_policy: OverlaySharingPolicy,
    pub max_pack_docs: NonZeroUsize,
    pub strategy: OverlayBatchPackingStrategy,
}

impl Default for OverlayTransportConfig {
    fn default() -> Self {
        Self {
            scope: OverlayDictionaryScope::BatchLocal,
            sharing_policy: OverlaySharingPolicy::default(),
            max_pack_docs: NonZeroUsize::new(16).unwrap_or(NonZeroUsize::MIN),
            strategy: OverlayBatchPackingStrategy::Sequential,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayTransportBatch {
    config: OverlayTransportConfig,
    pack: OverlayBatchPack,
}

impl OverlayTransportBatch {
    pub fn new(
        config: OverlayTransportConfig,
        pack: OverlayBatchPack,
    ) -> Result<Self, FractalError> {
        if !pack.exact_ok() {
            return Err(FractalError::InvalidState(
                "overlay transport batch must expand exactly back to canonical token ids"
                    .to_string(),
            ));
        }
        Ok(Self { config, pack })
    }

    pub fn config(&self) -> &OverlayTransportConfig {
        &self.config
    }

    pub fn pack(&self) -> &OverlayBatchPack {
        &self.pack
    }

    pub fn summary(&self) -> OverlayBatchPackSummary {
        self.pack.summary()
    }

    pub fn document_views(&self) -> &[PackedOverlayDocumentTransport] {
        &self.pack.document_views
    }

    pub fn exact_ok(&self) -> bool {
        self.pack.exact_ok()
    }
}

#[derive(Clone, Debug, Default)]
pub struct OverlayTransportAdapter {
    config: OverlayTransportConfig,
}

impl OverlayTransportAdapter {
    pub fn new(config: OverlayTransportConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &OverlayTransportConfig {
        &self.config
    }

    pub fn prepare_batch(
        &self,
        batch: &OverlayModelFacingBatch,
    ) -> Result<OverlayTransportBatch, FractalError> {
        let overlays = batch
            .documents()
            .iter()
            .map(|document| document.overlay().clone())
            .collect::<Vec<_>>();
        let pack = pack_overlay_documents_in_batches(
            self.config.scope,
            &overlays,
            &self.config.sharing_policy,
            self.config.max_pack_docs.get(),
            self.config.strategy,
        );
        OverlayTransportBatch::new(self.config.clone(), pack)
    }
}

impl ModelAdapter for OverlayTransportAdapter {
    type Input = OverlayModelFacingBatch;
    type Output = OverlayTransportBatch;

    fn prepare(&self, input: &Self::Input) -> Result<Self::Output, FractalError> {
        self.prepare_batch(input)
    }
}
