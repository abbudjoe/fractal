use fractal_core::error::FractalError;

use super::EncodedDocument;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceoffChunkLimits {
    pub max_tokens_per_chunk: usize,
    pub max_bytes_per_chunk: usize,
}

impl FaceoffChunkLimits {
    pub fn new(max_tokens_per_chunk: usize, max_bytes_per_chunk: usize) -> Self {
        Self {
            max_tokens_per_chunk,
            max_bytes_per_chunk,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceoffChunk {
    pub index: usize,
    pub token_count: usize,
    pub byte_count: usize,
    pub start: usize,
    pub end: usize,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceoffChunkedDocument {
    pub input_len: usize,
    pub frontier_token_count: usize,
    pub chunks: Vec<FaceoffChunk>,
}

impl FaceoffChunkedDocument {
    pub fn reconstruct(&self) -> Result<String, FractalError> {
        let mut ordered = self.chunks.iter().collect::<Vec<_>>();
        ordered.sort_by_key(|chunk| chunk.index);

        let mut expected_start = 0usize;
        let mut bytes = Vec::with_capacity(self.input_len);
        for chunk in ordered {
            if chunk.start != expected_start {
                return Err(FractalError::InvalidState(format!(
                    "chunk frontier has a gap/overlap at {}..{} (expected start {})",
                    chunk.start, chunk.end, expected_start
                )));
            }
            if chunk.payload.len() != chunk.byte_count {
                return Err(FractalError::InvalidState(format!(
                    "chunk payload length {} does not match byte_count {} for {}..{}",
                    chunk.payload.len(),
                    chunk.byte_count,
                    chunk.start,
                    chunk.end
                )));
            }
            expected_start = chunk.end;
            bytes.extend_from_slice(&chunk.payload);
        }

        if expected_start != self.input_len {
            return Err(FractalError::InvalidState(format!(
                "reconstructed length {} does not match input length {}",
                expected_start, self.input_len
            )));
        }
        if bytes.len() != self.input_len {
            return Err(FractalError::InvalidState(format!(
                "reconstructed byte count {} does not match input length {}",
                bytes.len(),
                self.input_len
            )));
        }

        String::from_utf8(bytes).map_err(|error| {
            FractalError::InvalidState(format!(
                "reconstructed chunk payload is not valid UTF-8: {error}"
            ))
        })
    }
}

pub(crate) fn package_document(
    document: &EncodedDocument,
    limits: FaceoffChunkLimits,
) -> Result<FaceoffChunkedDocument, FractalError> {
    if limits.max_tokens_per_chunk == 0 {
        return Err(FractalError::InvalidState(
            "max_tokens_per_chunk must be greater than zero".to_string(),
        ));
    }
    if limits.max_bytes_per_chunk == 0 {
        return Err(FractalError::InvalidState(
            "max_bytes_per_chunk must be greater than zero".to_string(),
        ));
    }

    let mut ordered = document.tokens.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|token| token.start);

    let mut chunks = Vec::new();
    let mut current_token_count = 0usize;
    let mut current_byte_count = 0usize;
    let mut current_start = 0usize;
    let mut current_end = 0usize;
    let mut current_payload = Vec::new();

    for token in ordered {
        if current_token_count == 0 {
            current_start = token.start;
        }
        current_end = token.end;
        current_token_count += 1;
        current_byte_count += token.bytes.len();
        current_payload.extend_from_slice(&token.bytes);

        let payload_is_utf8 = std::str::from_utf8(&current_payload).is_ok();
        let reached_token_limit = current_token_count >= limits.max_tokens_per_chunk;
        let reached_byte_limit = current_byte_count >= limits.max_bytes_per_chunk;

        if payload_is_utf8 && (reached_token_limit || reached_byte_limit) {
            chunks.push(FaceoffChunk {
                index: chunks.len(),
                token_count: current_token_count,
                byte_count: current_byte_count,
                start: current_start,
                end: current_end,
                payload: std::mem::take(&mut current_payload),
            });
            current_token_count = 0;
            current_byte_count = 0;
        }
    }

    if current_token_count > 0 {
        chunks.push(FaceoffChunk {
            index: chunks.len(),
            token_count: current_token_count,
            byte_count: current_byte_count,
            start: current_start,
            end: current_end,
            payload: current_payload,
        });
    }

    Ok(FaceoffChunkedDocument {
        input_len: document.input_len,
        frontier_token_count: document.tokens.len(),
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EncodedDocument, EncodedToken, EncodedTokenKind, FaceoffFallbackStats, FaceoffTokenId,
    };

    fn byte_token(id: u32, start: usize, end: usize, value: u8) -> EncodedToken {
        EncodedToken {
            id: FaceoffTokenId::new(id),
            kind: EncodedTokenKind::Byte { value },
            depth: 0,
            start,
            end,
            bytes: vec![value],
        }
    }

    #[test]
    fn packages_unicode_safe_chunk_boundaries() {
        let input = "🙂é語";
        let bytes = input.as_bytes();
        let tokens = bytes
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| byte_token(index as u32, index, index + 1, value))
            .collect::<Vec<_>>();
        let document = EncodedDocument {
            input_len: bytes.len(),
            tokens,
            fallback: FaceoffFallbackStats {
                motif_hits: 0,
                exact_motif_hits: 0,
                prototype_hits: 0,
                literal_hits: 0,
                shape_hits: 0,
                unknown_motifs: 0,
                recursed_to_children: 0,
                lexical_fallback_tokens: 0,
                byte_fallback_tokens: bytes.len(),
            },
        };

        let chunked = package_document(&document, FaceoffChunkLimits::new(1, 1)).unwrap();

        assert_eq!(chunked.reconstruct().unwrap(), input);
        assert!(chunked
            .chunks
            .iter()
            .all(|chunk| std::str::from_utf8(&chunk.payload).is_ok()));
        assert!(chunked.chunks.iter().all(|chunk| chunk.token_count >= 1));
    }
}
