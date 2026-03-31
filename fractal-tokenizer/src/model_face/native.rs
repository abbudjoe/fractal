use std::{error::Error, fmt, path::Path, str::Utf8Error};

use crate::{EncodedDocument, FaceoffChunk, FaceoffChunkedDocument};
use crate::{ModelFacingBatch, ModelFacingDocument};

/// Downstream tokenizer/retokenizer abstraction used by the native compatibility adapter.
pub trait NativeTokenizer {
    type Token;
    type Error;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error>;
}

/// HF-backed native tokenizer wrapper around `tokenizers::Tokenizer`.
#[derive(Clone)]
pub struct HuggingFaceNativeTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl HuggingFaceNativeTokenizer {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
    ) -> Result<Self, HuggingFaceNativeTokenizerError> {
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

        let mut expected_index = 0usize;
        let mut next_start = 0usize;
        let mut out = Vec::with_capacity(ordered.len());

        for chunk in ordered {
            if chunk.index != expected_index {
                return Err(NativeCompatibilityError::InconsistentDocument {
                    reason: format!(
                        "chunk indices are not contiguous: expected {} but saw {}",
                        expected_index, chunk.index
                    ),
                });
            }
            expected_index += 1;

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

impl NativeTokenizer for HuggingFaceNativeTokenizer {
    type Token = u32;
    type Error = HuggingFaceNativeTokenizerError;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|source| HuggingFaceNativeTokenizerError::Encode {
                reason: source.to_string(),
            })?;
        Ok(encoding.get_ids().to_vec())
    }
}
