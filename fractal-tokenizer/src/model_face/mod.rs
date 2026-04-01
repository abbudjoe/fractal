use fractal_core::error::FractalError;

use crate::faceoff::{EncodedDocument, FaceoffChunkLimits, FaceoffChunkedDocument};

mod batch;
mod bridge;
mod native;
mod traits;

pub use batch::ModelFacingBatch;
pub use bridge::{
    BridgeBatch, BridgeDocument, BridgeFeatureChunk, BridgeFeatureToken, EmbeddingBridgeAdapter,
    TypedEmbeddingBridge,
};
pub use native::{
    CanonicalPadSemantics, HuggingFaceNativeTokenizer, HuggingFaceNativeTokenizerError,
    ModelFacingDocumentView, NativeCollatedBatch, NativeCollatedChunk, NativeCollatedDocument,
    NativeCollationError, NativeCollationSpec, NativeCompatibilityAdapter,
    NativeCompatibilityBatch, NativeCompatibilityError, NativeTokenizedBatch, NativeTokenizedChunk,
    NativeTokenizedDocument, NativeTokenizer, NativeTruncationPolicy, SlowSentencePieceTokenizer,
    SlowSentencePieceTokenizerError,
};
pub use traits::{ModelAdapter, ModelBatch};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFacingDocument {
    encoded: EncodedDocument,
    chunked: FaceoffChunkedDocument,
}

impl ModelFacingDocument {
    pub fn new(
        encoded: EncodedDocument,
        chunked: FaceoffChunkedDocument,
    ) -> Result<Self, FractalError> {
        validate_model_facing_document(&encoded, &chunked)?;
        Ok(Self { encoded, chunked })
    }

    pub fn from_encoded(
        encoded: EncodedDocument,
        limits: FaceoffChunkLimits,
    ) -> Result<Self, FractalError> {
        let chunked = encoded.package(limits)?;
        Self::new(encoded, chunked)
    }

    pub fn encoded(&self) -> &EncodedDocument {
        &self.encoded
    }

    pub fn chunked(&self) -> &FaceoffChunkedDocument {
        &self.chunked
    }

    pub fn input_len(&self) -> usize {
        self.encoded.input_len
    }

    pub fn frontier_token_count(&self) -> usize {
        self.encoded.tokens.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.chunked.chunks.len()
    }

    pub fn fallback(&self) -> &crate::faceoff::FaceoffFallbackStats {
        &self.encoded.fallback
    }

    pub fn decode(&self) -> Result<String, FractalError> {
        self.encoded.decode()
    }

    pub fn reconstruct(&self) -> Result<String, FractalError> {
        self.chunked.reconstruct()
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        validate_model_facing_document(&self.encoded, &self.chunked)
    }

    pub fn into_parts(self) -> (EncodedDocument, FaceoffChunkedDocument) {
        (self.encoded, self.chunked)
    }
}

fn validate_model_facing_document(
    encoded: &EncodedDocument,
    chunked: &FaceoffChunkedDocument,
) -> Result<(), FractalError> {
    if encoded.input_len != chunked.input_len {
        return Err(FractalError::InvalidState(format!(
            "model-facing wrapper input length mismatch: encoded={} chunked={}",
            encoded.input_len, chunked.input_len
        )));
    }
    if encoded.tokens.len() != chunked.frontier_token_count {
        return Err(FractalError::InvalidState(format!(
            "model-facing wrapper frontier token count mismatch: encoded={} chunked={}",
            encoded.tokens.len(),
            chunked.frontier_token_count
        )));
    }

    let encoded_text = encoded.decode()?;
    let chunked_text = chunked.reconstruct()?;
    if encoded_text != chunked_text {
        return Err(FractalError::InvalidState(
            "model-facing wrapper encoded and chunked views reconstruct different text".to_string(),
        ));
    }

    Ok(())
}
