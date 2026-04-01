// Naming Convention for tokenizer primitives:
// [base]_[lever-description]_v[version]
// Examples: p1_fractal_hybrid_v1, b1_fractal_gated_dyn-residual-norm_v1
mod faceoff;
mod model_face;
mod primitives;
mod tokenizer;

pub use faceoff::{
    EncodedDocument, EncodedToken, EncodedTokenKind, FaceoffChunk, FaceoffChunkLimits,
    FaceoffChunkedDocument, FaceoffEmissionPolicy, FaceoffFallbackStats, FaceoffLexemeKind,
    FaceoffTokenId, FaceoffTokenizer, FaceoffVocab, FaceoffVocabConfig, ShapeEntry, VocabEntry,
    FACEOFF_VOCAB_FORMAT_VERSION,
};
pub use model_face::{
    BridgeBatch, BridgeDocument, BridgeFeatureChunk, BridgeFeatureToken, EmbeddingBridgeAdapter,
    HuggingFaceNativeTokenizer, HuggingFaceNativeTokenizerError, ModelAdapter, ModelBatch,
    ModelFacingBatch, ModelFacingDocument, NativeCollatedBatch, NativeCollatedChunk,
    NativeCollatedDocument, NativeCollationError, NativeCollationSpec, NativeCompatibilityAdapter,
    NativeCompatibilityBatch, NativeCompatibilityError, NativeTokenizedBatch, NativeTokenizedChunk,
    NativeTokenizedDocument, NativeTokenizer, NativeTruncationPolicy, TypedEmbeddingBridge,
};
pub use primitives::{
    B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot,
};
pub use tokenizer::{
    revived_primitive_factories, tokenizer_tracker_reminder, validate_tokenizer_primitive_name,
    PrimitiveFactory, PrimitiveRunSummary, RecursiveTokenizer, SplitPolicy, TokenRecord,
    TokenizerConfig,
};

#[cfg(test)]
mod tests;
