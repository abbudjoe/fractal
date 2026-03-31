use fractal_core::error::FractalError;

use crate::{
    EncodedTokenKind, FaceoffFallbackStats, FaceoffTokenId, ModelFacingBatch, ModelFacingDocument,
};

use super::{ModelAdapter, ModelBatch};

/// Per-token bridge metadata for model-facing adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeFeatureToken {
    pub document_index: usize,
    pub chunk_index: usize,
    pub token_index: usize,
    pub token_id: FaceoffTokenId,
    pub token_kind: EncodedTokenKind,
    pub depth: usize,
    pub start: usize,
    pub end: usize,
    pub span_len: usize,
}

/// Per-chunk bridge metadata for model-facing adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeFeatureChunk {
    pub document_index: usize,
    pub chunk_index: usize,
    pub start: usize,
    pub end: usize,
    pub token_count: usize,
    pub byte_count: usize,
    pub tokens: Vec<BridgeFeatureToken>,
}

/// Document-level bridge output that keeps structural metadata explicit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BridgeDocument {
    pub document_index: usize,
    pub input_len: usize,
    pub frontier_token_count: usize,
    pub fallback: FaceoffFallbackStats,
    pub chunks: Vec<BridgeFeatureChunk>,
}

impl BridgeDocument {
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn token_count(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.token_count).sum()
    }
}

/// Ordered bridge batch suitable for future embedding projection layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BridgeBatch {
    pub documents: Vec<BridgeDocument>,
}

impl BridgeBatch {
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn documents(&self) -> &[BridgeDocument] {
        &self.documents
    }

    pub fn iter(&self) -> impl Iterator<Item = &BridgeDocument> {
        self.documents.iter()
    }

    pub fn chunk_count(&self) -> usize {
        self.documents.iter().map(BridgeDocument::chunk_count).sum()
    }

    pub fn token_count(&self) -> usize {
        self.documents.iter().map(BridgeDocument::token_count).sum()
    }

    pub fn from_model_batch<B>(batch: &B) -> Result<Self, FractalError>
    where
        B: ModelBatch + ?Sized,
    {
        let mut documents = Vec::with_capacity(batch.len());
        for (document_index, document) in batch.documents().iter().enumerate() {
            documents.push(BridgeDocument::from_model_document(document_index, document)?);
        }
        Ok(Self { documents })
    }
}

/// Adapter trait for converting model-facing batches into bridge batches.
pub trait EmbeddingBridgeAdapter {
    fn bridge_batch<B>(&self, batch: &B) -> Result<BridgeBatch, FractalError>
    where
        B: ModelBatch + ?Sized;
}

/// Small concrete adapter that keeps the bridge layer model-family agnostic.
#[derive(Clone, Copy, Debug, Default)]
pub struct TypedEmbeddingBridge;

impl ModelAdapter for TypedEmbeddingBridge {
    type Input = ModelFacingBatch;
    type Output = BridgeBatch;

    fn prepare(&self, input: &Self::Input) -> Result<Self::Output, FractalError> {
        self.bridge_batch(input)
    }
}

impl EmbeddingBridgeAdapter for TypedEmbeddingBridge {
    fn bridge_batch<B>(&self, batch: &B) -> Result<BridgeBatch, FractalError>
    where
        B: ModelBatch + ?Sized,
    {
        BridgeBatch::from_model_batch(batch)
    }
}

impl BridgeDocument {
    pub fn from_model_document(
        document_index: usize,
        document: &ModelFacingDocument,
    ) -> Result<Self, FractalError> {
        document.validate()?;

        let encoded = document.encoded();
        let chunked = document.chunked();

        let mut ordered_tokens = encoded.tokens.iter().collect::<Vec<_>>();
        ordered_tokens.sort_by_key(|token| token.start);

        let mut ordered_chunks = chunked.chunks.iter().collect::<Vec<_>>();
        ordered_chunks.sort_by_key(|chunk| chunk.index);

        let mut token_cursor = 0usize;
        let mut bridge_chunks = Vec::with_capacity(ordered_chunks.len());

        for (chunk_position, chunk) in ordered_chunks.iter().enumerate() {
            if chunk.index != chunk_position {
                return Err(FractalError::InvalidState(format!(
                    "bridge chunk indices are not contiguous: expected {} but saw {}",
                    chunk_position, chunk.index
                )));
            }

            let mut tokens = Vec::with_capacity(chunk.token_count);
            while token_cursor < ordered_tokens.len() {
                let token = ordered_tokens[token_cursor];
                if token.start < chunk.start {
                    return Err(FractalError::InvalidState(format!(
                        "bridge token span {}..{} starts before chunk span {}..{}",
                        token.start, token.end, chunk.start, chunk.end
                    )));
                }
                if token.start >= chunk.end {
                    break;
                }
                if token.end > chunk.end {
                    return Err(FractalError::InvalidState(format!(
                        "bridge token span {}..{} exceeds chunk span {}..{}",
                        token.start, token.end, chunk.start, chunk.end
                    )));
                }

                tokens.push(BridgeFeatureToken {
                    document_index,
                    chunk_index: chunk.index,
                    token_index: token_cursor,
                    token_id: token.id,
                    token_kind: token.kind.clone(),
                    depth: token.depth,
                    start: token.start,
                    end: token.end,
                    span_len: token.end - token.start,
                });
                token_cursor += 1;
            }

            if tokens.len() != chunk.token_count {
                return Err(FractalError::InvalidState(format!(
                    "bridge token count {} does not match chunk token_count {} for chunk {}",
                    tokens.len(),
                    chunk.token_count,
                    chunk.index
                )));
            }

            bridge_chunks.push(BridgeFeatureChunk {
                document_index,
                chunk_index: chunk.index,
                start: chunk.start,
                end: chunk.end,
                token_count: chunk.token_count,
                byte_count: chunk.byte_count,
                tokens,
            });
        }

        if token_cursor != ordered_tokens.len() {
            return Err(FractalError::InvalidState(format!(
                "bridge consumed {} tokens but document contained {}",
                token_cursor,
                ordered_tokens.len()
            )));
        }

        Ok(Self {
            document_index,
            input_len: encoded.input_len,
            frontier_token_count: encoded.tokens.len(),
            fallback: encoded.fallback.clone(),
            chunks: bridge_chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncodedDocument, EncodedToken, FaceoffChunk, FaceoffChunkedDocument,
        FaceoffFallbackStats, FaceoffTokenId,
    };

    fn encoded_token(
        id: u32,
        _token_index: usize,
        depth: usize,
        start: usize,
        end: usize,
        kind: EncodedTokenKind,
        bytes: &[u8],
    ) -> EncodedToken {
        EncodedToken {
            id: FaceoffTokenId::new(id),
            kind,
            depth,
            start,
            end,
            bytes: bytes.to_vec(),
        }
    }

    fn sample_document(first_token_id: u32, input: &[u8]) -> ModelFacingDocument {
        let midpoint = input.len() / 2;
        let encoded = EncodedDocument {
            input_len: input.len(),
            tokens: vec![
                encoded_token(
                    first_token_id,
                    0,
                    0,
                    0,
                    midpoint,
                    EncodedTokenKind::Motif {
                        digest: format!("digest-{first_token_id}"),
                    },
                    &input[..midpoint],
                ),
                encoded_token(
                    first_token_id + 1,
                    1,
                    1,
                    midpoint,
                    input.len(),
                    EncodedTokenKind::Motif {
                        digest: format!("digest-{}", first_token_id + 1),
                    },
                    &input[midpoint..],
                ),
            ],
            fallback: FaceoffFallbackStats {
                motif_hits: 2,
                unknown_motifs: 0,
                recursed_to_children: 1,
                byte_fallback_tokens: 0,
            },
        };
        let chunked = FaceoffChunkedDocument {
            input_len: input.len(),
            frontier_token_count: encoded.tokens.len(),
            chunks: vec![
                FaceoffChunk {
                    index: 0,
                    token_count: 1,
                    byte_count: midpoint,
                    start: 0,
                    end: midpoint,
                    payload: input[..midpoint].to_vec(),
                },
                FaceoffChunk {
                    index: 1,
                    token_count: 1,
                    byte_count: input.len() - midpoint,
                    start: midpoint,
                    end: input.len(),
                    payload: input[midpoint..].to_vec(),
                },
            ],
        };
        ModelFacingDocument::new(encoded, chunked).unwrap()
    }

    fn sample_batch() -> ModelFacingBatch {
        ModelFacingBatch::from(vec![
            sample_document(1, b"abcd"),
            sample_document(10, b"wxyz"),
        ])
    }

    #[test]
    fn model_face_bridge_batch_from_document_preserves_order_and_spans() {
        let batch = sample_batch();
        let bridge = BridgeBatch::from_model_batch(&batch).unwrap();

        assert_eq!(bridge.len(), 2);
        assert_eq!(bridge.chunk_count(), 4);
        assert_eq!(bridge.documents[0].document_index, 0);
        assert_eq!(bridge.documents[1].document_index, 1);
        assert_eq!(bridge.documents[0].chunks[0].start, 0);
        assert_eq!(bridge.documents[0].chunks[0].end, 2);
        assert_eq!(bridge.documents[0].chunks[1].start, 2);
        assert_eq!(bridge.documents[0].chunks[1].end, 4);
        assert_eq!(bridge.documents[0].chunks[0].tokens[0].document_index, 0);
        assert_eq!(bridge.documents[0].chunks[0].tokens[0].chunk_index, 0);
        assert_eq!(bridge.documents[0].chunks[1].tokens[0].token_index, 1);
    }

    #[test]
    fn model_face_bridge_batch_preserves_structural_metadata() {
        let batch = sample_batch();
        let bridge = TypedEmbeddingBridge.bridge_batch(&batch).unwrap();
        let token = &bridge.documents[0].chunks[0].tokens[0];

        assert_eq!(token.token_id.as_u32(), 1);
        assert_eq!(
            token.token_kind,
            EncodedTokenKind::Motif {
                digest: "digest-1".to_string(),
            }
        );
        assert_eq!(token.depth, 0);
        assert_eq!(token.start, 0);
        assert_eq!(token.end, 2);
        assert_eq!(token.span_len, 2);
        assert_eq!(bridge.documents[0].fallback, FaceoffFallbackStats {
            motif_hits: 2,
            unknown_motifs: 0,
            recursed_to_children: 1,
            byte_fallback_tokens: 0,
        });
    }

    #[test]
    fn model_face_bridge_batch_is_deterministic_for_same_input() {
        let batch = sample_batch();
        let first = TypedEmbeddingBridge.bridge_batch(&batch).unwrap();
        let second = TypedEmbeddingBridge.bridge_batch(&batch).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn model_face_bridge_contract_is_model_family_agnostic() {
        fn assert_adapter<A: EmbeddingBridgeAdapter>() {}

        assert_adapter::<TypedEmbeddingBridge>();
    }
}
