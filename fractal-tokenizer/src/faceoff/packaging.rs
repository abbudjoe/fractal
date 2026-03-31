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
            FractalError::InvalidState(format!("reconstructed chunk payload is not valid UTF-8: {error}"))
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
        let would_exceed_tokens = current_token_count >= limits.max_tokens_per_chunk;
        let would_exceed_bytes = current_token_count > 0
            && current_byte_count + token.bytes.len() > limits.max_bytes_per_chunk;
        if would_exceed_tokens || would_exceed_bytes {
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

        if current_token_count == 0 {
            current_start = token.start;
        }
        current_end = token.end;
        current_token_count += 1;
        current_byte_count += token.bytes.len();
        current_payload.extend_from_slice(&token.bytes);
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
