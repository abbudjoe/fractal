use std::{
    error::Error,
    fmt,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    str::Utf8Error,
    sync::Arc,
};

use crate::{EncodedDocument, FaceoffChunk, FaceoffChunkedDocument};
use crate::{ModelFacingBatch, ModelFacingDocument};
use fractal_core::{data_generator::PAD_TOKEN, error::FractalError};
use sentencepiece::{SentencePieceError, SentencePieceProcessor};

/// Downstream tokenizer/retokenizer abstraction used by the native compatibility adapter.
pub trait NativeTokenizer {
    type Token;
    type Error;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error>;
}

/// Shared canonical pad-alias semantics for model training paths that must map
/// tokenizer-native padding onto the model's canonical pad token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalPadSemantics {
    pub native_pad_token_id: Option<u32>,
    pub canonical_model_pad_token_id: u32,
}

impl CanonicalPadSemantics {
    pub const fn canonical_default() -> Self {
        Self {
            native_pad_token_id: None,
            canonical_model_pad_token_id: PAD_TOKEN as u32,
        }
    }

    pub const fn new(native_pad_token_id: Option<u32>, canonical_model_pad_token_id: u32) -> Self {
        Self {
            native_pad_token_id,
            canonical_model_pad_token_id,
        }
    }

    pub fn from_expected_pad_token_id(
        expected_pad_token_id: usize,
        native_pad_token_id: Option<u32>,
    ) -> Result<Self, FractalError> {
        let canonical_model_pad_token_id = u32::try_from(expected_pad_token_id).map_err(|_| {
            FractalError::InvalidConfig(format!(
                "expected canonical pad token id {expected_pad_token_id} does not fit into u32"
            ))
        })?;
        Ok(Self::new(native_pad_token_id, canonical_model_pad_token_id))
    }

    pub fn collation_pad_token_id(&self) -> u32 {
        self.native_pad_token_id
            .unwrap_or(self.canonical_model_pad_token_id)
    }

    pub fn uses_canonical_pad_alias(&self) -> bool {
        self.native_pad_token_id != Some(self.canonical_model_pad_token_id)
    }

    pub fn canonicalize_input_from_chunk(&self, chunk: &NativeCollatedChunk<u32>) -> Vec<i64> {
        let valid_len = chunk.valid_token_count();
        chunk
            .padded_tokens
            .iter()
            .enumerate()
            .map(|(index, token)| {
                if index < valid_len {
                    *token as i64
                } else {
                    self.canonical_model_pad_token_id as i64
                }
            })
            .collect()
    }

    pub fn canonicalize_target_from_input(&self, input: &[i64], valid_len: usize) -> Vec<i64> {
        let mut target = vec![self.canonical_model_pad_token_id as i64; input.len()];
        if valid_len == 0 {
            return target;
        }
        let shifted_len = valid_len.saturating_sub(1);
        if shifted_len > 0 {
            target[..shifted_len].copy_from_slice(&input[1..(shifted_len + 1)]);
        }
        target[valid_len - 1] = self.canonical_model_pad_token_id as i64;
        target
    }
}

/// Shared slow SentencePiece wrapper used by Stage 0 and future model-facing
/// tokenization flows that require deterministic slow-tokenizer semantics.
#[derive(Clone, Debug)]
pub struct SlowSentencePieceTokenizer {
    processor: Arc<SentencePieceProcessor>,
}

impl SlowSentencePieceTokenizer {
    pub fn open(path: &Path) -> Result<Self, SlowSentencePieceTokenizerError> {
        let processor = SentencePieceProcessor::open(path).map_err(|source| {
            SlowSentencePieceTokenizerError::Load {
                path: path.to_path_buf(),
                reason: source,
            }
        })?;
        Ok(Self {
            processor: Arc::new(processor),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    pub fn model_pad_token_id(&self) -> Option<u32> {
        self.processor.pad_id()
    }
}

impl NativeTokenizer for SlowSentencePieceTokenizer {
    type Token = u32;
    type Error = SlowSentencePieceTokenizerError;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
        self.processor
            .encode(text)
            .map(|pieces| pieces.into_iter().map(|piece| piece.id).collect())
            .map_err(|source| SlowSentencePieceTokenizerError::Encode {
                input_preview: text.chars().take(32).collect(),
                reason: source,
            })
    }
}

#[derive(Debug)]
pub enum SlowSentencePieceTokenizerError {
    Load {
        path: PathBuf,
        reason: SentencePieceError,
    },
    Encode {
        input_preview: String,
        reason: SentencePieceError,
    },
}

impl fmt::Display for SlowSentencePieceTokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, reason } => {
                write!(
                    f,
                    "failed to load slow sentencepiece tokenizer from {}: {reason}",
                    path.display()
                )
            }
            Self::Encode {
                input_preview,
                reason,
            } => write!(
                f,
                "failed to encode text with slow sentencepiece tokenizer for input {:?}: {reason}",
                input_preview
            ),
        }
    }
}

impl Error for SlowSentencePieceTokenizerError {}

/// HF-backed native tokenizer wrapper around `tokenizers::Tokenizer`.
#[derive(Clone)]
pub struct HuggingFaceNativeTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceNativeTokenizer {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, HuggingFaceNativeTokenizerError> {
        tokenizers::Tokenizer::from_file(path.as_ref())
            .map(Self::new)
            .map_err(|source| HuggingFaceNativeTokenizerError::Load {
                reason: source.to_string(),
            })
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }
}

/// Errors produced by the HF-backed native tokenizer wrapper.
#[derive(Debug)]
pub enum HuggingFaceNativeTokenizerError {
    Load { reason: String },
    Encode { reason: String },
}

impl fmt::Display for HuggingFaceNativeTokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { reason } => write!(f, "failed to load tokenizer: {reason}"),
            Self::Encode { reason } => write!(f, "failed to encode text: {reason}"),
        }
    }
}

impl Error for HuggingFaceNativeTokenizerError {}

/// View trait used by adapter code that wants the combined wrapper contract.
///
/// `ModelFacingDocument` implements this trait directly so the adapter can
/// remain generic even while the phase is still stabilizing.
pub trait ModelFacingDocumentView {
    fn encoded(&self) -> &EncodedDocument;
    fn chunked(&self) -> &FaceoffChunkedDocument;
}

impl ModelFacingDocumentView for ModelFacingDocument {
    fn encoded(&self) -> &EncodedDocument {
        self.encoded()
    }

    fn chunked(&self) -> &FaceoffChunkedDocument {
        self.chunked()
    }
}

impl<T: ModelFacingDocumentView + ?Sized> ModelFacingDocumentView for &T {
    fn encoded(&self) -> &EncodedDocument {
        (**self).encoded()
    }

    fn chunked(&self) -> &FaceoffChunkedDocument {
        (**self).chunked()
    }
}

impl<T: ModelFacingDocumentView + ?Sized> ModelFacingDocumentView for Box<T> {
    fn encoded(&self) -> &EncodedDocument {
        (**self).encoded()
    }

    fn chunked(&self) -> &FaceoffChunkedDocument {
        (**self).chunked()
    }
}

/// A native-tokenized chunk that preserves the original faceoff chunk metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTokenizedChunk<T> {
    pub source: FaceoffChunk,
    pub native_tokens: Vec<T>,
}

/// The native-tokenized form of a packaged model-facing document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeTokenizedDocument<T> {
    pub input_len: usize,
    pub frontier_token_count: usize,
    pub chunks: Vec<NativeTokenizedChunk<T>>,
}

impl<T> NativeTokenizedDocument<T> {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn native_token_count(&self) -> usize {
        self.chunks
            .iter()
            .map(|chunk| chunk.native_tokens.len())
            .sum()
    }
}

/// Native-tokenized output for a batch of model-facing documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCompatibilityBatch<T> {
    pub documents: Vec<NativeTokenizedDocument<T>>,
}

impl<T> NativeCompatibilityBatch<T> {
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &NativeTokenizedDocument<T>> {
        self.documents.iter()
    }
}

/// Public alias matching the legacy re-export surface.
pub type NativeTokenizedBatch<T> = NativeCompatibilityBatch<T>;

/// Typed collation configuration for native-tokenized batches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCollationSpec<T> {
    pub pad_token: T,
    pub pad_to_multiple_of: Option<NonZeroUsize>,
    pub max_sequence_len: Option<NonZeroUsize>,
    pub truncation_policy: NativeTruncationPolicy,
}

/// Whether overflow should be rejected or truncated to the configured cap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeTruncationPolicy {
    #[default]
    Reject,
    Truncate,
}

impl<T> NativeCollationSpec<T> {
    pub fn new(pad_token: T) -> Self {
        Self {
            pad_token,
            pad_to_multiple_of: None,
            max_sequence_len: None,
            truncation_policy: NativeTruncationPolicy::Reject,
        }
    }

    pub fn try_new(
        pad_token: T,
        pad_to_multiple_of: Option<usize>,
    ) -> Result<Self, NativeCollationError> {
        let pad_to_multiple_of = match pad_to_multiple_of {
            Some(0) => {
                return Err(NativeCollationError::InvalidPadToMultipleOf { value: 0 });
            }
            Some(value) => Some(
                NonZeroUsize::new(value)
                    .ok_or(NativeCollationError::InvalidPadToMultipleOf { value })?,
            ),
            None => None,
        };

        Ok(Self {
            pad_token,
            pad_to_multiple_of,
            max_sequence_len: None,
            truncation_policy: NativeTruncationPolicy::Reject,
        })
    }

    pub fn with_pad_to_multiple_of(mut self, pad_to_multiple_of: NonZeroUsize) -> Self {
        self.pad_to_multiple_of = Some(pad_to_multiple_of);
        self
    }

    pub fn with_max_sequence_len(mut self, max_sequence_len: NonZeroUsize) -> Self {
        self.max_sequence_len = Some(max_sequence_len);
        self
    }

    pub fn with_truncation_policy(mut self, truncation_policy: NativeTruncationPolicy) -> Self {
        self.truncation_policy = truncation_policy;
        self
    }

    fn padded_sequence_len(&self, sequence_len: usize) -> usize {
        match self.pad_to_multiple_of {
            Some(multiple) => {
                let multiple = multiple.get();
                let remainder = sequence_len % multiple;
                if remainder == 0 {
                    sequence_len
                } else {
                    sequence_len + (multiple - remainder)
                }
            }
            None => sequence_len,
        }
    }
}

/// Errors produced while padding and collating native-tokenized batches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCollationError {
    InvalidPadToMultipleOf {
        value: usize,
    },
    InconsistentChunkOrder {
        document_index: usize,
        chunk_index: usize,
        source_index: usize,
    },
    SequenceTooLong {
        document_index: usize,
        chunk_index: usize,
        actual_len: usize,
        max_sequence_len: usize,
    },
}

impl fmt::Display for NativeCollationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPadToMultipleOf { value } => {
                write!(f, "pad_to_multiple_of must be greater than zero, got {value}")
            }
            Self::InconsistentChunkOrder {
                document_index,
                chunk_index,
                source_index,
            } => write!(
                f,
                "document {document_index} chunk {chunk_index} is out of source order: saw source index {source_index}"
            ),
            Self::SequenceTooLong {
                document_index,
                chunk_index,
                actual_len,
                max_sequence_len,
            } => write!(
                f,
                "document {document_index} chunk {chunk_index} is too long: actual length {actual_len} exceeds max sequence length {max_sequence_len}"
            ),
        }
    }
}

impl Error for NativeCollationError {}

/// Padded native-tokenized chunk with deterministic attention masking metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCollatedChunk<T> {
    pub source_document_index: usize,
    pub source_chunk_index: usize,
    pub source: NativeTokenizedChunk<T>,
    pub padded_tokens: Vec<T>,
    pub attention_mask: Vec<bool>,
}

impl<T> NativeCollatedChunk<T> {
    pub fn valid_token_count(&self) -> usize {
        self.attention_mask.iter().filter(|mask| **mask).count()
    }

    pub fn padded_token_count(&self) -> usize {
        self.padded_tokens.len()
    }
}

/// Document-level grouping for collated native-tokenized chunks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCollatedDocument<T> {
    pub source_document_index: usize,
    pub input_len: usize,
    pub frontier_token_count: usize,
    pub chunks: Vec<NativeCollatedChunk<T>>,
}

impl<T> NativeCollatedDocument<T> {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn padded_token_count(&self) -> usize {
        self.chunks
            .first()
            .map(NativeCollatedChunk::padded_token_count)
            .unwrap_or(0)
    }

    pub fn native_token_count(&self) -> usize {
        self.chunks
            .iter()
            .map(NativeCollatedChunk::valid_token_count)
            .sum()
    }
}

/// Collated native-tokenized batch with deterministic padding and ordering metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeCollatedBatch<T> {
    pub spec: NativeCollationSpec<T>,
    pub sequence_len: usize,
    pub documents: Vec<NativeCollatedDocument<T>>,
}

impl<T> NativeCollatedBatch<T> {
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &NativeCollatedDocument<T>> {
        self.documents.iter()
    }

    pub fn chunks(&self) -> impl Iterator<Item = &NativeCollatedChunk<T>> {
        self.documents
            .iter()
            .flat_map(|document| document.chunks.iter())
    }

    pub fn chunk_count(&self) -> usize {
        self.documents
            .iter()
            .map(NativeCollatedDocument::chunk_count)
            .sum()
    }
}

/// Errors produced by the native compatibility adapter.
#[derive(Debug)]
pub enum NativeCompatibilityError<E> {
    InconsistentDocument {
        reason: String,
    },
    InvalidChunkUtf8 {
        chunk_index: usize,
        source: Utf8Error,
    },
    Tokenizer(E),
}

impl<E: fmt::Display> fmt::Display for NativeCompatibilityError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InconsistentDocument { reason } => {
                write!(f, "model-facing document is inconsistent: {reason}")
            }
            Self::InvalidChunkUtf8 {
                chunk_index,
                source,
            } => {
                write!(
                    f,
                    "chunk {chunk_index} payload is not valid UTF-8: {source}"
                )
            }
            Self::Tokenizer(error) => write!(f, "native tokenizer failed: {error}"),
        }
    }
}

impl<E> Error for NativeCompatibilityError<E> where E: fmt::Debug + fmt::Display + 'static {}

/// Stateless adapter that retokenizes packaged frontier chunks using a native tokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeCompatibilityAdapter;

impl NativeCompatibilityAdapter {
    pub fn retokenize_document<D, N>(
        &self,
        document: &D,
        tokenizer: &N,
    ) -> Result<NativeTokenizedDocument<N::Token>, NativeCompatibilityError<N::Error>>
    where
        D: ModelFacingDocumentView,
        N: NativeTokenizer,
    {
        let encoded = document.encoded();
        let chunked = document.chunked();

        if encoded.input_len != chunked.input_len {
            return Err(NativeCompatibilityError::InconsistentDocument {
                reason: format!(
                    "encoded input length {} does not match chunked input length {}",
                    encoded.input_len, chunked.input_len
                ),
            });
        }
        if encoded.tokens.len() != chunked.frontier_token_count {
            return Err(NativeCompatibilityError::InconsistentDocument {
                reason: format!(
                    "encoded frontier token count {} does not match chunked frontier token count {}",
                    encoded.tokens.len(),
                    chunked.frontier_token_count
                ),
            });
        }

        let mut ordered = chunked.chunks.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|chunk| chunk.index);

        let mut next_start = 0usize;
        let mut out = Vec::with_capacity(ordered.len());

        for (expected_index, chunk) in ordered.into_iter().enumerate() {
            if chunk.index != expected_index {
                return Err(NativeCompatibilityError::InconsistentDocument {
                    reason: format!(
                        "chunk indices are not contiguous: expected {} but saw {}",
                        expected_index, chunk.index
                    ),
                });
            }

            if chunk.start != next_start {
                return Err(NativeCompatibilityError::InconsistentDocument {
                    reason: format!(
                        "chunk frontier has a gap/overlap at {}..{} (expected start {})",
                        chunk.start, chunk.end, next_start
                    ),
                });
            }
            if chunk.payload.len() != chunk.byte_count {
                return Err(NativeCompatibilityError::InconsistentDocument {
                    reason: format!(
                        "chunk payload length {} does not match byte_count {} for {}..{}",
                        chunk.payload.len(),
                        chunk.byte_count,
                        chunk.start,
                        chunk.end
                    ),
                });
            }

            let chunk_text = std::str::from_utf8(&chunk.payload).map_err(|source| {
                NativeCompatibilityError::InvalidChunkUtf8 {
                    chunk_index: chunk.index,
                    source,
                }
            })?;
            let native_tokens = tokenizer
                .tokenize(chunk_text)
                .map_err(NativeCompatibilityError::Tokenizer)?;

            next_start = chunk.end;
            out.push(NativeTokenizedChunk {
                source: chunk.clone(),
                native_tokens,
            });
        }

        if next_start != encoded.input_len {
            return Err(NativeCompatibilityError::InconsistentDocument {
                reason: format!(
                    "reconstructed chunk span end {} does not match input length {}",
                    next_start, encoded.input_len
                ),
            });
        }

        Ok(NativeTokenizedDocument {
            input_len: encoded.input_len,
            frontier_token_count: chunked.frontier_token_count,
            chunks: out,
        })
    }

    pub fn retokenize_batch<N>(
        &self,
        batch: &ModelFacingBatch,
        tokenizer: &N,
    ) -> Result<NativeCompatibilityBatch<N::Token>, NativeCompatibilityError<N::Error>>
    where
        N: NativeTokenizer,
    {
        let mut documents = Vec::with_capacity(batch.len());
        for document in batch.iter() {
            documents.push(self.retokenize_document(document, tokenizer)?);
        }
        Ok(NativeCompatibilityBatch { documents })
    }
}

impl<T: Clone> NativeCompatibilityBatch<T> {
    pub fn collate(
        &self,
        spec: &NativeCollationSpec<T>,
    ) -> Result<NativeCollatedBatch<T>, NativeCollationError> {
        let max_sequence_len = self
            .documents
            .iter()
            .flat_map(|document| document.chunks.iter())
            .map(|chunk| chunk.native_tokens.len())
            .max()
            .unwrap_or(0);
        let mut sequence_len = spec.padded_sequence_len(max_sequence_len);

        if let Some(max_sequence_len) = spec.max_sequence_len {
            let max_sequence_len = max_sequence_len.get();
            if sequence_len > max_sequence_len {
                match spec.truncation_policy {
                    NativeTruncationPolicy::Reject => {
                        for (document_index, document) in self.documents.iter().enumerate() {
                            for (chunk_index, chunk) in document.chunks.iter().enumerate() {
                                let padded_len =
                                    spec.padded_sequence_len(chunk.native_tokens.len());
                                if padded_len > max_sequence_len {
                                    return Err(NativeCollationError::SequenceTooLong {
                                        document_index,
                                        chunk_index,
                                        actual_len: padded_len,
                                        max_sequence_len,
                                    });
                                }
                            }
                        }
                        unreachable!(
                            "at least one chunk must exceed the configured max when the batch max does"
                        );
                    }
                    NativeTruncationPolicy::Truncate => {
                        sequence_len = max_sequence_len;
                    }
                }
            }
        }

        let mut documents = Vec::with_capacity(self.documents.len());
        for (document_index, document) in self.documents.iter().enumerate() {
            let mut chunks = Vec::with_capacity(document.chunks.len());
            for (chunk_index, chunk) in document.chunks.iter().enumerate() {
                if chunk.source.index != chunk_index {
                    return Err(NativeCollationError::InconsistentChunkOrder {
                        document_index,
                        chunk_index,
                        source_index: chunk.source.index,
                    });
                }

                let valid_len = std::cmp::min(chunk.native_tokens.len(), sequence_len);
                let mut padded_tokens = chunk.native_tokens[..valid_len].to_vec();
                padded_tokens.resize(sequence_len, spec.pad_token.clone());

                let mut attention_mask = vec![true; valid_len];
                attention_mask.resize(sequence_len, false);

                chunks.push(NativeCollatedChunk {
                    source_document_index: document_index,
                    source_chunk_index: chunk_index,
                    source: chunk.clone(),
                    padded_tokens,
                    attention_mask,
                });
            }

            documents.push(NativeCollatedDocument {
                source_document_index: document_index,
                input_len: document.input_len,
                frontier_token_count: document.frontier_token_count,
                chunks,
            });
        }

        Ok(NativeCollatedBatch {
            spec: spec.clone(),
            sequence_len,
            documents,
        })
    }
}

impl NativeTokenizer for HuggingFaceNativeTokenizer {
    type Token = u32;
    type Error = HuggingFaceNativeTokenizerError;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
        let encoding = self.tokenizer.encode(text, false).map_err(|source| {
            HuggingFaceNativeTokenizerError::Encode {
                reason: source.to_string(),
            }
        })?;
        Ok(encoding.get_ids().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, process::Command};

    fn sentencepiece_testdata_model() -> PathBuf {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("cargo metadata should run");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata should be valid json");
        let packages = metadata["packages"]
            .as_array()
            .expect("cargo metadata should include packages");
        let manifest_path = packages
            .iter()
            .find(|package| package["name"] == "sentencepiece")
            .and_then(|package| package["manifest_path"].as_str())
            .expect("sentencepiece package should be present in cargo metadata");
        PathBuf::from(manifest_path)
            .parent()
            .expect("sentencepiece manifest should have parent")
            .join("testdata")
            .join("toy.model")
    }

    fn chunk(
        index: usize,
        token_count: usize,
        start: usize,
        end: usize,
        native_tokens: Vec<u32>,
    ) -> NativeTokenizedChunk<u32> {
        NativeTokenizedChunk {
            source: FaceoffChunk {
                index,
                token_count,
                byte_count: end - start,
                start,
                end,
                payload: vec![b'x'; end - start],
            },
            native_tokens,
        }
    }

    #[test]
    fn collates_batches_with_padding_masks_and_source_order() {
        let batch = NativeCompatibilityBatch {
            documents: vec![
                NativeTokenizedDocument {
                    input_len: 5,
                    frontier_token_count: 3,
                    chunks: vec![chunk(0, 2, 0, 2, vec![10, 11]), chunk(1, 1, 2, 5, vec![12])],
                },
                NativeTokenizedDocument {
                    input_len: 4,
                    frontier_token_count: 2,
                    chunks: vec![chunk(0, 2, 0, 4, vec![20, 21, 22])],
                },
            ],
        };
        let spec = NativeCollationSpec::new(0u32)
            .with_pad_to_multiple_of(NonZeroUsize::new(4).expect("non-zero literal"));

        let collated = batch.collate(&spec).expect("collation succeeds");

        assert_eq!(collated.len(), 2);
        assert_eq!(collated.chunk_count(), 3);
        assert_eq!(collated.sequence_len, 4);
        assert_eq!(collated.spec, spec);

        let document0 = &collated.documents[0];
        assert_eq!(document0.source_document_index, 0);
        assert_eq!(document0.chunk_count(), 2);
        assert_eq!(document0.padded_token_count(), 4);
        assert_eq!(document0.chunks[0].source_document_index, 0);
        assert_eq!(document0.chunks[0].source_chunk_index, 0);
        assert_eq!(document0.chunks[0].padded_tokens, vec![10, 11, 0, 0]);
        assert_eq!(
            document0.chunks[0].attention_mask,
            vec![true, true, false, false]
        );
        assert_eq!(document0.chunks[1].source_chunk_index, 1);
        assert_eq!(document0.chunks[1].padded_tokens, vec![12, 0, 0, 0]);
        assert_eq!(
            document0.chunks[1].attention_mask,
            vec![true, false, false, false]
        );

        let document1 = &collated.documents[1];
        assert_eq!(document1.source_document_index, 1);
        assert_eq!(document1.chunk_count(), 1);
        assert_eq!(document1.chunks[0].padded_tokens, vec![20, 21, 22, 0]);
        assert_eq!(
            document1.chunks[0].attention_mask,
            vec![true, true, true, false]
        );
    }

    #[test]
    fn rejects_invalid_pad_multiple_and_chunk_order_mismatches() {
        assert!(matches!(
            NativeCollationSpec::try_new(0u32, Some(0)),
            Err(NativeCollationError::InvalidPadToMultipleOf { value: 0 })
        ));

        let batch = NativeCompatibilityBatch {
            documents: vec![NativeTokenizedDocument {
                input_len: 2,
                frontier_token_count: 1,
                chunks: vec![chunk(1, 1, 0, 2, vec![1])],
            }],
        };
        let spec = NativeCollationSpec::new(0u32);

        assert!(matches!(
            batch.collate(&spec),
            Err(NativeCollationError::InconsistentChunkOrder {
                document_index: 0,
                chunk_index: 0,
                source_index: 1,
            })
        ));
    }

    #[test]
    fn collates_empty_batch_stably() {
        let batch = NativeCompatibilityBatch::<u32> { documents: vec![] };
        let spec = NativeCollationSpec::new(0u32);

        let collated = batch.collate(&spec).expect("empty batch should collate");

        assert!(collated.is_empty());
        assert_eq!(collated.len(), 0);
        assert_eq!(collated.chunk_count(), 0);
        assert_eq!(collated.sequence_len, 0);
    }

    #[test]
    fn rejects_overflow_without_truncation_policy() {
        let batch = NativeCompatibilityBatch {
            documents: vec![NativeTokenizedDocument {
                input_len: 3,
                frontier_token_count: 1,
                chunks: vec![chunk(0, 1, 0, 3, vec![1, 2, 3])],
            }],
        };
        let spec = NativeCollationSpec::new(0u32)
            .with_max_sequence_len(NonZeroUsize::new(2).expect("non-zero literal"));

        assert!(matches!(
            batch.collate(&spec),
            Err(NativeCollationError::SequenceTooLong {
                document_index: 0,
                chunk_index: 0,
                actual_len: 3,
                max_sequence_len: 2,
            })
        ));
    }

    #[test]
    fn truncates_with_explicit_policy() {
        let batch = NativeCompatibilityBatch {
            documents: vec![NativeTokenizedDocument {
                input_len: 3,
                frontier_token_count: 1,
                chunks: vec![chunk(0, 1, 0, 3, vec![1, 2, 3])],
            }],
        };
        let spec = NativeCollationSpec::new(0u32)
            .with_max_sequence_len(NonZeroUsize::new(2).expect("non-zero literal"))
            .with_truncation_policy(NativeTruncationPolicy::Truncate);

        let collated = batch.collate(&spec).expect("truncation should collate");

        assert_eq!(collated.sequence_len, 2);
        assert_eq!(collated.len(), 1);
        assert_eq!(collated.documents[0].chunks[0].valid_token_count(), 2);
        assert_eq!(collated.documents[0].chunks[0].padded_token_count(), 2);
        assert_eq!(collated.documents[0].chunks[0].padded_tokens, vec![1, 2]);
        assert_eq!(
            collated.documents[0].chunks[0].attention_mask,
            vec![true, true]
        );
    }

    #[test]
    fn multi_document_order_is_stable_under_truncation() {
        let batch = NativeCompatibilityBatch {
            documents: vec![
                NativeTokenizedDocument {
                    input_len: 3,
                    frontier_token_count: 1,
                    chunks: vec![chunk(0, 1, 0, 3, vec![1, 2, 3])],
                },
                NativeTokenizedDocument {
                    input_len: 2,
                    frontier_token_count: 1,
                    chunks: vec![chunk(0, 1, 0, 2, vec![4, 5])],
                },
                NativeTokenizedDocument {
                    input_len: 4,
                    frontier_token_count: 1,
                    chunks: vec![chunk(0, 1, 0, 4, vec![6, 7, 8, 9])],
                },
            ],
        };
        let spec = NativeCollationSpec::new(0u32)
            .with_max_sequence_len(NonZeroUsize::new(2).expect("non-zero literal"))
            .with_truncation_policy(NativeTruncationPolicy::Truncate);

        let collated = batch.collate(&spec).expect("truncation should collate");

        assert_eq!(
            collated
                .documents
                .iter()
                .map(|document| document.source_document_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            collated
                .documents
                .iter()
                .map(|document| document.chunks[0].source_chunk_index)
                .collect::<Vec<_>>(),
            vec![0, 0, 0]
        );
        assert_eq!(
            collated
                .documents
                .iter()
                .map(|document| document.chunks[0].valid_token_count())
                .collect::<Vec<_>>(),
            vec![2, 2, 2]
        );
        assert_eq!(collated.sequence_len, 2);
    }

    #[test]
    fn canonical_pad_semantics_alias_missing_native_pad_tokens() {
        let semantics = CanonicalPadSemantics::from_expected_pad_token_id(PAD_TOKEN, None)
            .expect("canonical pad semantics should build");
        let chunk = NativeCollatedChunk {
            source_document_index: 0,
            source_chunk_index: 0,
            padded_tokens: vec![10, 11, 777],
            attention_mask: vec![true, true, false],
            source: NativeTokenizedChunk {
                source: FaceoffChunk {
                    index: 0,
                    token_count: 2,
                    byte_count: 3,
                    start: 0,
                    end: 3,
                    payload: vec![b'x'; 3],
                },
                native_tokens: vec![10, 11],
            },
        };

        let input = semantics.canonicalize_input_from_chunk(&chunk);
        let target = semantics.canonicalize_target_from_input(&input, chunk.valid_token_count());

        assert_eq!(semantics.collation_pad_token_id(), PAD_TOKEN as u32);
        assert!(semantics.uses_canonical_pad_alias());
        assert_eq!(input, vec![10, 11, PAD_TOKEN as i64]);
        assert_eq!(target, vec![11, PAD_TOKEN as i64, PAD_TOKEN as i64]);
    }

    #[test]
    fn slow_sentencepiece_tokenizer_loads_test_model() {
        let tokenizer = SlowSentencePieceTokenizer::open(&sentencepiece_testdata_model())
            .expect("test sentencepiece model should load");

        let tokens = tokenizer
            .tokenize("hello world")
            .expect("slow sentencepiece tokenizer should encode");

        assert!(tokenizer.vocab_size() > 0);
        assert!(!tokens.is_empty());
    }
}
