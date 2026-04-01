// Naming Convention for tokenizer primitives:
// [base]_[lever-description]_v[version]
// Examples: p1_fractal_hybrid_v1, b1_fractal_gated_dyn-residual-norm_v1
mod faceoff;
mod model_face;
mod overlay;
mod primitives;
mod tokenizer;

pub use faceoff::{
    EncodedDocument, EncodedToken, EncodedTokenKind, FaceoffChunk, FaceoffChunkLimits,
    FaceoffChunkedDocument, FaceoffEmissionPolicy, FaceoffEncodingOptions, FaceoffFallbackMode,
    FaceoffFallbackStats, FaceoffIdentityMode, FaceoffLexemeKind, FaceoffLocalCacheMode,
    FaceoffTokenId, FaceoffTokenizer, FaceoffVocab, FaceoffVocabConfig, PrototypeAdmissionPolicy,
    PrototypeEntry, PrototypeGranularityMode, ShapeEntry, VocabEntry, FACEOFF_VOCAB_FORMAT_VERSION,
};
pub use model_face::{
    BridgeBatch, BridgeDocument, BridgeFeatureChunk, BridgeFeatureToken, CanonicalPadSemantics,
    CanonicalTokenization,
    EmbeddingBridgeAdapter, HuggingFaceNativeTokenizer, HuggingFaceNativeTokenizerError,
    ModelAdapter, ModelBatch, ModelFacingBatch, ModelFacingDocument, NativeCollatedBatch,
    NativeCollatedChunk, NativeCollatedDocument, NativeCollationError, NativeCollationSpec,
    NativeCompatibilityAdapter, NativeCompatibilityBatch, NativeCompatibilityError,
    NativeTokenizedBatch, NativeTokenizedChunk, NativeTokenizedDocument, NativeTokenizer,
    NativeTruncationPolicy, SlowSentencePieceTokenizer, SlowSentencePieceTokenizerError,
    TypedEmbeddingBridge,
};
pub use overlay::{
    build_recursive_overlay, pack_overlay_documents_in_batches, LocalMacro, LocalMacroKind,
    OverlayBatchPack, OverlayBatchPackSummary, OverlayBatchPackingStrategy, OverlayDictionaryScope,
    OverlayDocumentMode, OverlayPack, OverlaySegment, OverlaySharingPolicy,
    OverlayTransportSummary, PackedOverlayDocument, PackedOverlayDocumentTransport,
    PackedOverlaySegment, RecursiveOverlayConfig, RecursiveOverlayDocument, RecursiveOverlayMode,
    SharedFactor, SharedMacro, SharedMacroDefinitionSegment,
};
pub use primitives::{
    B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot,
};
pub use tokenizer::{
    p1_dynamic_lever_factory, revived_primitive_factories, tokenizer_tracker_reminder,
    validate_tokenizer_primitive_name, MotifReusePolicy, PrimitiveFactory, PrimitiveRunSummary,
    RecursiveTokenizer, SplitPolicy, StateSignature, TokenRecord, TokenizerConfig,
    TokenizerSubstrateMode,
};

#[cfg(test)]
mod tests;
