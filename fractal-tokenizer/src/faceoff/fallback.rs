use std::collections::BTreeMap;

use fractal_core::error::FractalError;

use crate::{PrimitiveRunSummary, TokenRecord};

use super::{
    lexeme::scan_lexemes, vocab::token_digest, EncodedDocument, EncodedToken, EncodedTokenKind,
    FaceoffEmissionPolicy, FaceoffFallbackMode, FaceoffIdentityMode, FaceoffLocalCacheMode,
    FaceoffTokenId, FaceoffVocab,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FaceoffFallbackStats {
    pub motif_hits: usize,
    pub exact_motif_hits: usize,
    pub prototype_hits: usize,
    pub literal_hits: usize,
    pub shape_hits: usize,
    pub unknown_motifs: usize,
    pub recursed_to_children: usize,
    pub local_cache_hits: usize,
    pub local_cache_stores: usize,
    pub lexical_fallback_tokens: usize,
    pub byte_fallback_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LocalSpanKey {
    depth: usize,
    text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LocalMotifEntry {
    id: FaceoffTokenId,
    digest: String,
}

#[derive(Default)]
struct LocalMotifCache {
    entries: BTreeMap<LocalSpanKey, LocalMotifEntry>,
    next_ordinal: u32,
}

struct EncodeNodeContext<'a> {
    tree: &'a SummaryTree<'a>,
    input: &'a [u8],
    vocab: &'a FaceoffVocab,
    policy: FaceoffEmissionPolicy,
    fallback_mode: FaceoffFallbackMode,
    local_cache_mode: FaceoffLocalCacheMode,
}

#[derive(Default)]
struct EncodeNodeState {
    encoded: Vec<EncodedToken>,
    stats: FaceoffFallbackStats,
    local_cache: LocalMotifCache,
}

impl LocalMotifCache {
    fn lookup(&self, depth: usize, text: &str) -> Option<&LocalMotifEntry> {
        self.entries.get(&LocalSpanKey {
            depth,
            text: text.to_owned(),
        })
    }

    fn store(&mut self, vocab: &FaceoffVocab, depth: usize, text: &str, digest: &str) -> bool {
        let key = LocalSpanKey {
            depth,
            text: text.to_owned(),
        };
        if self.entries.contains_key(&key) {
            return false;
        }

        let entry = LocalMotifEntry {
            id: vocab.local_motif_id(self.next_ordinal),
            digest: format!("local:d{depth}:{digest}"),
        };
        self.next_ordinal += 1;
        self.entries.insert(key, entry);
        true
    }
}

pub(crate) fn encode_summary_document(
    text: &str,
    summary: &PrimitiveRunSummary,
    vocab: &FaceoffVocab,
    policy: FaceoffEmissionPolicy,
    fallback_mode: FaceoffFallbackMode,
    local_cache_mode: FaceoffLocalCacheMode,
) -> Result<EncodedDocument, FractalError> {
    let input = text.as_bytes();
    let tree = SummaryTree::new(summary, input.len())?;
    let root = tree.root()?;
    let context = EncodeNodeContext {
        tree: &tree,
        input,
        vocab,
        policy,
        fallback_mode,
        local_cache_mode,
    };
    let mut state = EncodeNodeState::default();
    encode_node(root, &context, &mut state)?;
    state.encoded.sort_by_key(|token| token.start);
    Ok(EncodedDocument {
        input_len: input.len(),
        tokens: state.encoded,
        fallback: state.stats,
    })
}

fn encode_node(
    record: &TokenRecord,
    context: &EncodeNodeContext<'_>,
    state: &mut EncodeNodeState,
) -> Result<(), FractalError> {
    if record.end > context.input.len() || record.start > record.end {
        return Err(FractalError::InvalidState(format!(
            "token span {}..{} is out of bounds for input length {}",
            record.start,
            record.end,
            context.input.len()
        )));
    }

    let digest = token_digest(&record.token)?;
    let children = context.tree.children(record);
    let children_cover_parent = spans_cover_parent(record, &children);
    let should_recurse_known = should_recurse_known(
        record,
        &children,
        children_cover_parent,
        context.policy,
        &context.tree.digest_counts,
        context.tree.max_depth,
    );
    if context.vocab.identity_mode() == FaceoffIdentityMode::PrototypePrimary {
        if let Some(id) = context.vocab.prototype_id(record)? {
            if should_recurse_known {
                state.stats.recursed_to_children += 1;
                for child in &children {
                    encode_node(child, context, state)?;
                }
                return Ok(());
            }

            state.stats.motif_hits += 1;
            state.stats.prototype_hits += 1;
            state.encoded.push(EncodedToken {
                id,
                kind: EncodedTokenKind::Motif {
                    digest: context.vocab.motif_digest(id).unwrap_or(digest).to_owned(),
                },
                depth: record.depth,
                start: record.start,
                end: record.end,
                bytes: context.input[record.start..record.end].to_vec(),
            });
            return Ok(());
        }
    } else if let Some(id) = context.vocab.motif_id(digest) {
        if should_recurse_known {
            state.stats.recursed_to_children += 1;
            for child in &children {
                encode_node(child, context, state)?;
            }
            return Ok(());
        }

        state.stats.motif_hits += 1;
        state.stats.exact_motif_hits += 1;
        state.encoded.push(EncodedToken {
            id,
            kind: EncodedTokenKind::Motif {
                digest: context.vocab.motif_digest(id).unwrap_or(digest).to_owned(),
            },
            depth: record.depth,
            start: record.start,
            end: record.end,
            bytes: context.input[record.start..record.end].to_vec(),
        });
        return Ok(());
    }

    if context.vocab.identity_mode() == FaceoffIdentityMode::Legacy {
        if let Some(id) = context.vocab.prototype_id(record)? {
            if should_recurse_known {
                state.stats.recursed_to_children += 1;
                for child in &children {
                    encode_node(child, context, state)?;
                }
                return Ok(());
            }

            state.stats.motif_hits += 1;
            state.stats.prototype_hits += 1;
            state.encoded.push(EncodedToken {
                id,
                kind: EncodedTokenKind::Motif {
                    digest: context.vocab.motif_digest(id).unwrap_or(digest).to_owned(),
                },
                depth: record.depth,
                start: record.start,
                end: record.end,
                bytes: context.input[record.start..record.end].to_vec(),
            });
            return Ok(());
        }
    }

    let literal =
        std::str::from_utf8(&context.input[record.start..record.end]).map_err(|source| {
            FractalError::InvalidState(format!(
                "faceoff fallback span {}..{} is not valid UTF-8: {source}",
                record.start, record.end
            ))
        })?;
    if context.vocab.identity_mode() == FaceoffIdentityMode::Legacy
        && context.fallback_mode.allows_literal_rescue()
    {
        if let Some(id) = context.vocab.literal_id(literal) {
            state.stats.motif_hits += 1;
            state.stats.literal_hits += 1;
            state.encoded.push(EncodedToken {
                id,
                kind: EncodedTokenKind::Motif {
                    digest: context.vocab.motif_digest(id).unwrap_or(digest).to_owned(),
                },
                depth: record.depth,
                start: record.start,
                end: record.end,
                bytes: context.input[record.start..record.end].to_vec(),
            });
            return Ok(());
        }
    }

    if context.vocab.identity_mode() == FaceoffIdentityMode::Legacy
        && context.fallback_mode.allows_shape_rescue()
    {
        if let Some(id) = context.vocab.shape_id_for_text(literal) {
            if should_recurse_known {
                state.stats.recursed_to_children += 1;
                for child in &children {
                    encode_node(child, context, state)?;
                }
                return Ok(());
            }

            state.stats.motif_hits += 1;
            state.stats.shape_hits += 1;
            state.encoded.push(EncodedToken {
                id,
                kind: EncodedTokenKind::Motif {
                    digest: context.vocab.motif_digest(id).unwrap_or(digest).to_owned(),
                },
                depth: record.depth,
                start: record.start,
                end: record.end,
                bytes: context.input[record.start..record.end].to_vec(),
            });
            return Ok(());
        }
    }

    if context.local_cache_mode == FaceoffLocalCacheMode::ExactSpan {
        if let Some(entry) = state.local_cache.lookup(record.depth, literal) {
            state.stats.motif_hits += 1;
            state.stats.local_cache_hits += 1;
            state.encoded.push(EncodedToken {
                id: entry.id,
                kind: EncodedTokenKind::Motif {
                    digest: entry.digest.clone(),
                },
                depth: record.depth,
                start: record.start,
                end: record.end,
                bytes: context.input[record.start..record.end].to_vec(),
            });
            return Ok(());
        }
    }

    state.stats.unknown_motifs += 1;
    if children_cover_parent {
        state.stats.recursed_to_children += 1;
        for child in &children {
            encode_node(child, context, state)?;
        }
        maybe_store_local_cache_entry(
            context.local_cache_mode,
            &mut state.local_cache,
            context.vocab,
            record,
            literal,
            digest,
            &mut state.stats,
        );
        return Ok(());
    }

    emit_lexical_or_byte_fallback(
        record,
        context.input,
        context.vocab,
        &mut state.encoded,
        &mut state.stats,
    );
    maybe_store_local_cache_entry(
        context.local_cache_mode,
        &mut state.local_cache,
        context.vocab,
        record,
        literal,
        digest,
        &mut state.stats,
    );
    Ok(())
}

fn maybe_store_local_cache_entry(
    local_cache_mode: FaceoffLocalCacheMode,
    local_cache: &mut LocalMotifCache,
    vocab: &FaceoffVocab,
    record: &TokenRecord,
    literal: &str,
    digest: &str,
    stats: &mut FaceoffFallbackStats,
) {
    if local_cache_mode != FaceoffLocalCacheMode::ExactSpan {
        return;
    }
    if local_cache.store(vocab, record.depth, literal, digest) {
        stats.local_cache_stores += 1;
    }
}

fn emit_lexical_or_byte_fallback(
    record: &TokenRecord,
    input: &[u8],
    vocab: &FaceoffVocab,
    encoded: &mut Vec<EncodedToken>,
    stats: &mut FaceoffFallbackStats,
) {
    let span = &input[record.start..record.end];
    if let Ok(text) = std::str::from_utf8(span) {
        for lexeme in scan_lexemes(text, record.start) {
            encoded.push(EncodedToken {
                id: vocab.lexical_id(lexeme.kind),
                kind: EncodedTokenKind::Lexical { kind: lexeme.kind },
                depth: record.depth,
                start: lexeme.start,
                end: lexeme.end,
                bytes: lexeme.bytes,
            });
            stats.lexical_fallback_tokens += 1;
        }
        return;
    }

    for (offset, value) in span.iter().copied().enumerate() {
        let start = record.start + offset;
        let end = start + 1;
        encoded.push(EncodedToken {
            id: vocab.byte_id(value),
            kind: EncodedTokenKind::Byte { value },
            depth: record.depth,
            start,
            end,
            bytes: vec![value],
        });
        stats.byte_fallback_tokens += 1;
    }
}

fn should_recurse_known(
    parent: &TokenRecord,
    children: &[&TokenRecord],
    children_cover_parent: bool,
    policy: FaceoffEmissionPolicy,
    digest_counts: &BTreeMap<String, usize>,
    max_depth: usize,
) -> bool {
    match policy {
        FaceoffEmissionPolicy::GreedyKnown => false,
        FaceoffEmissionPolicy::FinestKnown => {
            children_cover_parent && children.iter().any(|child| child.end > child.start)
        }
        FaceoffEmissionPolicy::StateAware => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let Some(parent_bin) = state_bin(parent) else {
                return true;
            };
            let child_bins = children
                .iter()
                .filter_map(|child| state_bin(child))
                .collect::<Vec<_>>();
            if child_bins.len() != children.len() {
                return true;
            }

            let child_peak = child_bins.iter().copied().max().unwrap_or(parent_bin);
            let child_mean_num = child_bins.iter().map(|&bin| u32::from(bin)).sum::<u32>();
            let child_mean_den = child_bins.len() as u32;

            child_peak > parent_bin && child_mean_num > u32::from(parent_bin) * child_mean_den
        }
        FaceoffEmissionPolicy::ReuseAware => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let parent_reuse = motif_reuse_count(parent, digest_counts).unwrap_or(0);
            if parent_reuse <= 1 {
                return true;
            }

            let child_reuse_counts = children
                .iter()
                .filter_map(|child| motif_reuse_count(child, digest_counts))
                .collect::<Vec<_>>();
            if child_reuse_counts.is_empty() {
                return true;
            }

            let child_peak = child_reuse_counts.iter().copied().max().unwrap_or(0);
            let child_mean_num = child_reuse_counts.iter().sum::<usize>();
            let child_mean_den = child_reuse_counts.len();

            child_peak > parent_reuse || child_mean_num > parent_reuse * child_mean_den
        }
        FaceoffEmissionPolicy::NoveltyAware => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let parent_reuse = motif_reuse_count(parent, digest_counts).unwrap_or(0);
            let child_reuse_counts = children
                .iter()
                .filter_map(|child| motif_reuse_count(child, digest_counts))
                .collect::<Vec<_>>();
            if child_reuse_counts.is_empty() {
                return false;
            }

            let child_peak = child_reuse_counts.iter().copied().max().unwrap_or(0);
            let child_mean_num = child_reuse_counts.iter().sum::<usize>();
            let child_mean_den = child_reuse_counts.len();

            if parent_reuse <= 1 {
                !(child_peak > 1 || child_mean_num > child_mean_den)
            } else {
                child_peak > parent_reuse || child_mean_num > parent_reuse * child_mean_den
            }
        }
        FaceoffEmissionPolicy::HybridStructural => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let parent_reuse = motif_reuse_count(parent, digest_counts).unwrap_or(0);
            let child_reuse_counts = children
                .iter()
                .filter_map(|child| motif_reuse_count(child, digest_counts))
                .collect::<Vec<_>>();
            if child_reuse_counts.is_empty() {
                return false;
            }

            let child_lengths = children
                .iter()
                .map(|child| child.end.saturating_sub(child.start))
                .collect::<Vec<_>>();
            if child_lengths.is_empty() {
                return false;
            }

            let child_peak_reuse = child_reuse_counts.iter().copied().max().unwrap_or(0);
            let child_mean_reuse_num = child_reuse_counts.iter().sum::<usize>();
            let child_mean_reuse_den = child_reuse_counts.len();
            let child_peak_len = child_lengths.iter().copied().max().unwrap_or(0);
            let child_mean_len_num = child_lengths.iter().sum::<usize>();
            let child_mean_len_den = child_lengths.len();
            let parent_len = parent.end.saturating_sub(parent.start);

            let reuse_signal = if parent_reuse <= 1 {
                child_peak_reuse > 1 || child_mean_reuse_num > child_mean_reuse_den
            } else {
                child_peak_reuse > parent_reuse
                    || child_mean_reuse_num > parent_reuse * child_mean_reuse_den
            };
            let span_signal = child_peak_len.saturating_mul(2) > parent_len
                || child_mean_len_num.saturating_mul(2)
                    > parent_len.saturating_mul(child_mean_len_den);

            reuse_signal && span_signal
        }
        FaceoffEmissionPolicy::SpanLengthAware => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let parent_len = parent.end.saturating_sub(parent.start);
            let child_lengths = children
                .iter()
                .map(|child| child.end.saturating_sub(child.start))
                .collect::<Vec<_>>();
            if child_lengths.is_empty() {
                return false;
            }

            let child_peak = child_lengths.iter().copied().max().unwrap_or(0);
            let child_mean_num = child_lengths.iter().sum::<usize>();
            let child_mean_den = child_lengths.len();

            let children_are_tight = child_peak.saturating_mul(2) <= parent_len
                && child_mean_num.saturating_mul(2) <= parent_len.saturating_mul(child_mean_den);

            !children_are_tight
        }
        FaceoffEmissionPolicy::Budgeted => {
            if !children_cover_parent || children.is_empty() {
                return false;
            }
            if !children.iter().any(|child| child.end > child.start) {
                return false;
            }

            let remaining_budget = max_depth.saturating_sub(parent.depth);
            if remaining_budget == 0 {
                return false;
            }

            let frontier_cost = children.len();

            frontier_cost <= remaining_budget
        }
    }
}

fn spans_cover_parent(parent: &TokenRecord, children: &[&TokenRecord]) -> bool {
    if children.is_empty() {
        return false;
    }

    let mut expected_start = parent.start;
    for child in children {
        if child.start != expected_start || child.start > child.end {
            return false;
        }
        expected_start = child.end;
    }
    expected_start == parent.end
}

fn state_bin(record: &TokenRecord) -> Option<u16> {
    record
        .token
        .split('-')
        .find_map(|part| part.strip_prefix('q'))
        .and_then(|value| value.parse::<u16>().ok())
}

struct SummaryTree<'a> {
    by_depth: BTreeMap<usize, Vec<&'a TokenRecord>>,
    digest_counts: BTreeMap<String, usize>,
    max_depth: usize,
    root: &'a TokenRecord,
}

impl<'a> SummaryTree<'a> {
    fn new(summary: &'a PrimitiveRunSummary, input_len: usize) -> Result<Self, FractalError> {
        if summary.tokens.is_empty() {
            return Err(FractalError::InvalidState(
                "cannot build face-off tree from an empty token summary".to_owned(),
            ));
        }
        let mut by_depth = BTreeMap::<usize, Vec<&TokenRecord>>::new();
        let mut digest_counts = BTreeMap::<String, usize>::new();
        for token in &summary.tokens {
            by_depth.entry(token.depth).or_default().push(token);
            let digest = token_digest(&token.token).map(str::to_owned)?;
            *digest_counts.entry(digest).or_default() += 1;
        }
        for layer in by_depth.values_mut() {
            layer.sort_by_key(|record| record.start);
        }

        let max_depth = by_depth.keys().copied().max().unwrap_or(0);

        let root = by_depth
            .get(&0)
            .and_then(|layer| {
                layer
                    .iter()
                    .copied()
                    .find(|record| record.start == 0 && record.end == input_len)
            })
            .ok_or_else(|| {
                FractalError::InvalidState(format!(
                    "no root token (depth=0, span=0..{input_len}) found in summary"
                ))
            })?;
        Ok(Self {
            by_depth,
            digest_counts,
            max_depth,
            root,
        })
    }

    fn root(&self) -> Result<&'a TokenRecord, FractalError> {
        if self.root.start > self.root.end {
            return Err(FractalError::InvalidState(
                "root token has inverted span".to_owned(),
            ));
        }
        Ok(self.root)
    }

    fn children(&self, parent: &TokenRecord) -> Vec<&'a TokenRecord> {
        let next_depth = parent.depth + 1;
        self.by_depth
            .get(&next_depth)
            .map(|layer| {
                layer
                    .iter()
                    .copied()
                    .filter(|record| record.start >= parent.start && record.end <= parent.end)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }
}

fn motif_reuse_count(
    record: &TokenRecord,
    digest_counts: &BTreeMap<String, usize>,
) -> Option<usize> {
    let digest = token_digest(&record.token).ok()?;
    digest_counts.get(digest).copied()
}

#[cfg(test)]
mod tests {
    use crate::FaceoffLexemeKind;

    use super::super::lexeme::scan_lexemes;

    #[test]
    fn scan_lexemes_is_deterministic_for_typed_ascii_atoms() {
        let input = "AuthProvider 2026-03-31 ::git-push{ x }\n    next_line";

        let first = scan_lexemes(input, 0);
        let second = scan_lexemes(input, 0);

        assert_eq!(first, second);
        assert!(first
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Identifier));
        assert!(first
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Number));
        assert!(first
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::Whitespace));
        assert!(first
            .iter()
            .any(|span| span.kind == FaceoffLexemeKind::NewlineIndent));
    }

    #[test]
    fn scan_lexemes_preserves_utf8_boundaries() {
        let input = "こんにちは 世界\n🙂 signal";
        let spans = scan_lexemes(input, 0);

        assert_eq!(
            spans.iter().map(|span| span.bytes.len()).sum::<usize>(),
            input.len()
        );
        for span in spans {
            let text = std::str::from_utf8(&span.bytes).unwrap();
            assert!(!text.is_empty());
        }
    }
}
