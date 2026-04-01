use super::FaceoffLexemeKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LexemeSpan {
    pub(crate) kind: FaceoffLexemeKind,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) bytes: Vec<u8>,
}

pub(crate) fn scan_lexemes(text: &str, absolute_start: usize) -> Vec<LexemeSpan> {
    let mut spans = Vec::new();
    let mut offset = 0usize;

    while offset < text.len() {
        let next = consume_lexeme(text, offset);
        spans.push(LexemeSpan {
            kind: next.0,
            start: absolute_start + offset,
            end: absolute_start + next.1,
            bytes: text.as_bytes()[offset..next.1].to_vec(),
        });
        offset = next.1;
    }

    spans
}

pub(crate) fn lexical_shape_key(text: &str) -> String {
    let fragments = scan_lexemes(text, 0)
        .into_iter()
        .map(|span| match span.kind {
            FaceoffLexemeKind::Word => "word".to_string(),
            FaceoffLexemeKind::Identifier => "identifier".to_string(),
            FaceoffLexemeKind::Number => "number".to_string(),
            FaceoffLexemeKind::Whitespace => "whitespace".to_string(),
            FaceoffLexemeKind::NewlineIndent => {
                let indent = span.bytes.len().saturating_sub(1);
                format!("newline_indent:{indent}")
            }
            FaceoffLexemeKind::Punctuation => {
                format!("punct:{}:{}", span.bytes.len(), bytes_to_hex(&span.bytes))
            }
            FaceoffLexemeKind::SymbolRun => {
                format!("symbol:{}:{}", span.bytes.len(), bytes_to_hex(&span.bytes))
            }
        })
        .collect::<Vec<_>>();
    fragments.join("\u{001f}")
}

fn consume_lexeme(text: &str, start: usize) -> (FaceoffLexemeKind, usize) {
    let ch = text[start..].chars().next().unwrap_or('\0');
    if ch == '\n' {
        return (
            FaceoffLexemeKind::NewlineIndent,
            consume_newline_indent(text, start),
        );
    }
    if ch.is_whitespace() {
        return (
            FaceoffLexemeKind::Whitespace,
            consume_while(text, start, |value| value.is_whitespace() && value != '\n'),
        );
    }
    if ch.is_ascii_digit() {
        return (
            FaceoffLexemeKind::Number,
            consume_while(text, start, |value| value.is_ascii_digit()),
        );
    }
    if is_identifier_start(ch) {
        let end = consume_while(text, start, is_identifier_continue);
        let kind = if text[start..end]
            .chars()
            .all(|value| value.is_alphabetic() && !value.is_uppercase())
        {
            FaceoffLexemeKind::Word
        } else {
            FaceoffLexemeKind::Identifier
        };
        return (kind, end);
    }
    if is_punctuation(ch) {
        return (
            FaceoffLexemeKind::Punctuation,
            consume_while(text, start, is_punctuation),
        );
    }
    (
        FaceoffLexemeKind::SymbolRun,
        consume_while(text, start, |value| {
            !value.is_whitespace() && !value.is_alphanumeric() && !is_punctuation(value)
        }),
    )
}

fn consume_newline_indent(text: &str, start: usize) -> usize {
    let after_newline = start + '\n'.len_utf8();
    consume_while(text, after_newline, |value| matches!(value, ' ' | '\t'))
}

fn consume_while(text: &str, start: usize, predicate: impl Fn(char) -> bool) -> usize {
    let mut end = start;
    for (offset, ch) in text[start..].char_indices() {
        if !predicate(ch) {
            break;
        }
        end = start + offset + ch.len_utf8();
    }
    end.max(
        start
            + text[start..]
                .chars()
                .next()
                .map(char::len_utf8)
                .unwrap_or(0),
    )
}

fn is_identifier_start(value: char) -> bool {
    value.is_alphabetic() || value == '_'
}

fn is_identifier_continue(value: char) -> bool {
    value.is_alphanumeric() || value == '_'
}

fn is_punctuation(value: char) -> bool {
    matches!(
        value,
        '.' | ',' | ';' | ':' | '!' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\''
    )
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(hex_digit(byte >> 4));
        out.push(hex_digit(byte & 0x0f));
    }
    out
}

fn hex_digit(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + (value - 10)) as char,
        _ => '0',
    }
}

#[cfg(test)]
mod tests {
    use super::{lexical_shape_key, scan_lexemes};
    use crate::FaceoffLexemeKind;

    #[test]
    fn lexical_shape_key_abstracts_identifiers_numbers_and_words() {
        let left = lexical_shape_key("fn render_home() {\n    let auth_provider = 2026;\n}\n");
        let right = lexical_shape_key("fn render_settings() {\n    let oauth_flow = 2027;\n}\n");
        assert_eq!(left, right);
    }

    #[test]
    fn scan_lexemes_classifies_expected_atoms() {
        let spans = scan_lexemes("AuthProvider 2026 ::git-push{ x }\n    next_line", 0);

        assert!(spans
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Identifier));
        assert!(spans
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Number));
        assert!(spans
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Whitespace));
        assert!(spans
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::NewlineIndent));
    }
}
