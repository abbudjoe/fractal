use fractal_core::error::FractalError;

use super::{EncodedDocument, EncodedToken};

pub(crate) fn decode_document(document: &EncodedDocument) -> Result<String, FractalError> {
    let ordered = validated_tokens(document)?;

    let mut output = Vec::with_capacity(document.input_len);
    for token in ordered {
        output.extend_from_slice(&token.bytes);
    }

    String::from_utf8(output).map_err(|error| {
        FractalError::InvalidState(format!("decoded document is not valid UTF-8: {error}"))
    })
}

pub(crate) fn validated_tokens(
    document: &EncodedDocument,
) -> Result<Vec<&EncodedToken>, FractalError> {
    let mut ordered = document.tokens.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|token| token.start);

    let mut expected_start = 0usize;
    let mut output_len = 0usize;
    for token in &ordered {
        if token.start != expected_start {
            return Err(FractalError::InvalidState(format!(
                "encoded token stream has a span gap/overlap at {}..{} (expected start {})",
                token.start, token.end, expected_start
            )));
        }
        if token.start > token.end {
            return Err(FractalError::InvalidState(format!(
                "encoded token has inverted span {}..{}",
                token.start, token.end
            )));
        }
        let span_len = token.end - token.start;
        if token.bytes.len() != span_len {
            return Err(FractalError::InvalidState(format!(
                "encoded token bytes length {} does not match span length {} for {}..{}",
                token.bytes.len(),
                span_len,
                token.start,
                token.end
            )));
        }
        output_len += token.bytes.len();
        expected_start = token.end;
    }

    if expected_start != document.input_len {
        return Err(FractalError::InvalidState(format!(
            "decoded length {} does not match document input length {}",
            expected_start, document.input_len
        )));
    }
    if output_len != document.input_len {
        return Err(FractalError::InvalidState(format!(
            "decoded byte count {} does not match document input length {}",
            output_len, document.input_len
        )));
    }

    Ok(ordered)
}
