use fractal_core::error::FractalError;

mod decode;
mod encode;
mod fallback;
mod packaging;
mod vocab;

pub use encode::FaceoffTokenizer;
pub use fallback::FaceoffFallbackStats;
pub use packaging::{FaceoffChunk, FaceoffChunkLimits, FaceoffChunkedDocument};
pub use vocab::{FaceoffVocab, FaceoffVocabConfig, VocabEntry, FACEOFF_VOCAB_FORMAT_VERSION};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceoffEmissionPolicy {
    GreedyKnown,
    FinestKnown,
    StateAware,
    ReuseAware,
    NoveltyAware,
    HybridStructural,
    SpanLengthAware,
    Budgeted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceoffTokenId(u32);

impl FaceoffTokenId {
    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FaceoffLexemeKind {
    Word,
    Identifier,
    Number,
    Whitespace,
    NewlineIndent,
    Punctuation,
    SymbolRun,
}

impl FaceoffLexemeKind {
    pub(crate) fn stable_index(self) -> u32 {
        match self {
            Self::Word => 0,
            Self::Identifier => 1,
            Self::Number => 2,
            Self::Whitespace => 3,
            Self::NewlineIndent => 4,
            Self::Punctuation => 5,
            Self::SymbolRun => 6,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodedTokenKind {
    Motif { digest: String },
    Lexical { kind: FaceoffLexemeKind },
    Byte { value: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedToken {
    pub id: FaceoffTokenId,
    pub kind: EncodedTokenKind,
    pub depth: usize,
    pub start: usize,
    pub end: usize,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedDocument {
    pub input_len: usize,
    pub tokens: Vec<EncodedToken>,
    pub fallback: FaceoffFallbackStats,
}

impl EncodedDocument {
    pub fn decode(&self) -> Result<String, FractalError> {
        decode::decode_document(self)
    }

    pub fn package(
        &self,
        limits: FaceoffChunkLimits,
    ) -> Result<FaceoffChunkedDocument, FractalError> {
        packaging::package_document(self, limits)
    }
}
