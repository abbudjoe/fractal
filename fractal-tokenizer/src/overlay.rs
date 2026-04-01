use std::collections::BTreeMap;

use crate::CanonicalTokenization;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecursiveOverlayMode {
    #[default]
    Off,
    LocalLineMacro,
}

impl RecursiveOverlayMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::LocalLineMacro => "local-line-macro",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveOverlayConfig {
    pub min_repeat_count: usize,
    pub min_span_tokens: usize,
    pub max_span_tokens: usize,
    pub min_repeated_token_mass: usize,
}

impl Default for RecursiveOverlayConfig {
    fn default() -> Self {
        Self {
            min_repeat_count: 2,
            min_span_tokens: 4,
            max_span_tokens: 128,
            min_repeated_token_mass: 16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverlayDocumentMode {
    Passthrough,
    LocalMacro,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalMacroKind {
    RepeatedLine,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalMacro {
    pub macro_id: usize,
    pub kind: LocalMacroKind,
    pub token_ids: Vec<u32>,
    pub use_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OverlaySegment {
    BaseSlice { start: usize, len: usize },
    MacroRef { macro_id: usize, span_len: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveOverlayDocument {
    pub canonical: CanonicalTokenization,
    pub mode: OverlayDocumentMode,
    pub macros: Vec<LocalMacro>,
    pub segments: Vec<OverlaySegment>,
}

impl RecursiveOverlayDocument {
    pub fn passthrough(canonical: CanonicalTokenization) -> Self {
        let len = canonical.token_ids.len();
        let segments = if len == 0 {
            Vec::new()
        } else {
            vec![OverlaySegment::BaseSlice { start: 0, len }]
        };
        Self {
            canonical,
            mode: OverlayDocumentMode::Passthrough,
            macros: Vec::new(),
            segments,
        }
    }

    pub fn expand_token_ids(&self) -> Vec<u32> {
        let macro_map = self
            .macros
            .iter()
            .map(|entry| (entry.macro_id, entry.token_ids.as_slice()))
            .collect::<BTreeMap<_, _>>();
        let mut expanded = Vec::with_capacity(self.canonical.token_ids.len());
        for segment in &self.segments {
            match segment {
                OverlaySegment::BaseSlice { start, len } => {
                    expanded.extend_from_slice(&self.canonical.token_ids[*start..(*start + *len)]);
                }
                OverlaySegment::MacroRef { macro_id, .. } => {
                    if let Some(tokens) = macro_map.get(macro_id) {
                        expanded.extend_from_slice(tokens);
                    }
                }
            }
        }
        expanded
    }

    pub fn exact_ok(&self) -> bool {
        self.expand_token_ids() == self.canonical.token_ids
    }

    pub fn macro_ref_count(&self) -> usize {
        self.segments
            .iter()
            .filter(|segment| matches!(segment, OverlaySegment::MacroRef { .. }))
            .count()
    }

    pub fn overlay_symbol_count(&self) -> usize {
        let base_slice_cost = self
            .segments
            .iter()
            .map(|segment| match segment {
                OverlaySegment::BaseSlice { len, .. } => *len,
                OverlaySegment::MacroRef { .. } => 1,
            })
            .sum::<usize>();
        let macro_def_cost = self
            .macros
            .iter()
            .map(|entry| entry.token_ids.len())
            .sum::<usize>();
        base_slice_cost + macro_def_cost
    }

    pub fn repeated_token_mass_saved(&self) -> usize {
        self.canonical
            .token_ids
            .len()
            .saturating_sub(self.overlay_symbol_count())
    }

    pub fn compression_ratio_vs_canonical(&self) -> f64 {
        let cost = self.overlay_symbol_count().max(1);
        self.canonical.token_ids.len() as f64 / cost as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LineTokenSpan {
    token_start: usize,
    token_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MacroCandidate {
    token_ids: Vec<u32>,
    count: usize,
    saved_mass: usize,
}

pub fn build_recursive_overlay(
    text: &str,
    canonical: CanonicalTokenization,
    mode: RecursiveOverlayMode,
    config: &RecursiveOverlayConfig,
) -> RecursiveOverlayDocument {
    match mode {
        RecursiveOverlayMode::Off => RecursiveOverlayDocument::passthrough(canonical),
        RecursiveOverlayMode::LocalLineMacro => {
            build_local_line_macro_overlay(text, canonical, config)
        }
    }
}

fn build_local_line_macro_overlay(
    text: &str,
    canonical: CanonicalTokenization,
    config: &RecursiveOverlayConfig,
) -> RecursiveOverlayDocument {
    let line_spans = line_token_spans(text, &canonical.offsets);
    let macro_catalog = select_line_macros(&canonical.token_ids, &line_spans, config);
    if macro_catalog.is_empty() {
        return RecursiveOverlayDocument::passthrough(canonical);
    }

    let mut segments = Vec::new();
    let mut cursor = 0usize;
    let mut used_macro_ids = BTreeMap::<usize, usize>::new();
    for span in line_spans.into_iter().flatten() {
        if span.token_end <= span.token_start {
            continue;
        }

        if cursor < span.token_start {
            push_base_slice(&mut segments, cursor, span.token_start - cursor);
        }

        let line_tokens = &canonical.token_ids[span.token_start..span.token_end];
        if let Some(macro_entry) = macro_catalog.get(line_tokens) {
            segments.push(OverlaySegment::MacroRef {
                macro_id: macro_entry.macro_id,
                span_len: line_tokens.len(),
            });
            *used_macro_ids.entry(macro_entry.macro_id).or_default() += 1;
        } else {
            push_base_slice(
                &mut segments,
                span.token_start,
                span.token_end - span.token_start,
            );
        }
        cursor = span.token_end;
    }

    if cursor < canonical.token_ids.len() {
        push_base_slice(&mut segments, cursor, canonical.token_ids.len() - cursor);
    }

    let macros = macro_catalog
        .values()
        .filter_map(|entry| {
            used_macro_ids
                .get(&entry.macro_id)
                .map(|use_count| LocalMacro {
                    macro_id: entry.macro_id,
                    kind: LocalMacroKind::RepeatedLine,
                    token_ids: entry.token_ids.clone(),
                    use_count: *use_count,
                })
        })
        .collect::<Vec<_>>();

    if macros.is_empty() {
        return RecursiveOverlayDocument::passthrough(canonical);
    }

    RecursiveOverlayDocument {
        canonical,
        mode: OverlayDocumentMode::LocalMacro,
        macros,
        segments,
    }
}

fn select_line_macros(
    token_ids: &[u32],
    line_spans: &[Option<LineTokenSpan>],
    config: &RecursiveOverlayConfig,
) -> BTreeMap<Vec<u32>, LocalMacro> {
    let mut candidates = BTreeMap::<Vec<u32>, usize>::new();
    for span in line_spans.iter().flatten() {
        let len = span.token_end.saturating_sub(span.token_start);
        if len < config.min_span_tokens || len > config.max_span_tokens {
            continue;
        }
        let slice = token_ids[span.token_start..span.token_end].to_vec();
        *candidates.entry(slice).or_default() += 1;
    }

    let mut selected = candidates
        .into_iter()
        .filter_map(|(token_ids, count)| {
            if count < config.min_repeat_count {
                return None;
            }
            let saved_mass = token_ids.len().saturating_mul(count.saturating_sub(1));
            if saved_mass < config.min_repeated_token_mass {
                return None;
            }
            Some(MacroCandidate {
                token_ids,
                count,
                saved_mass,
            })
        })
        .collect::<Vec<_>>();

    selected.sort_by(|left, right| {
        right
            .saved_mass
            .cmp(&left.saved_mass)
            .then_with(|| right.token_ids.len().cmp(&left.token_ids.len()))
            .then_with(|| left.token_ids.cmp(&right.token_ids))
    });

    selected
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| {
            let token_ids = candidate.token_ids;
            let entry = LocalMacro {
                macro_id: index,
                kind: LocalMacroKind::RepeatedLine,
                token_ids: token_ids.clone(),
                use_count: candidate.count,
            };
            (token_ids, entry)
        })
        .collect()
}

fn line_token_spans(text: &str, offsets: &[(usize, usize)]) -> Vec<Option<LineTokenSpan>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for line in text.split_inclusive('\n') {
        let end = start + line.len();
        ranges.push((start, end));
        start = end;
    }
    if text.is_empty() {
        ranges.push((0, 0));
    }

    let mut spans = Vec::with_capacity(ranges.len());
    let mut cursor = 0usize;
    for (_, line_end) in ranges {
        let token_start = cursor;
        while cursor < offsets.len() && offsets[cursor].0 < line_end {
            cursor += 1;
        }
        spans.push(if cursor > token_start {
            Some(LineTokenSpan {
                token_start,
                token_end: cursor,
            })
        } else {
            None
        });
    }

    spans
}

fn push_base_slice(segments: &mut Vec<OverlaySegment>, start: usize, len: usize) {
    if len == 0 {
        return;
    }
    if let Some(OverlaySegment::BaseSlice {
        start: last_start,
        len: last_len,
    }) = segments.last_mut()
    {
        if *last_start + *last_len == start {
            *last_len += len;
            return;
        }
    }
    segments.push(OverlaySegment::BaseSlice { start, len });
}
