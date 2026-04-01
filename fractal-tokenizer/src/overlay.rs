use std::collections::BTreeMap;

use crate::CanonicalTokenization;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RecursiveOverlayMode {
    #[default]
    Off,
    LocalLineMacro,
    LocalRecordMacro,
}

impl RecursiveOverlayMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::LocalLineMacro => "local-line-macro",
            Self::LocalRecordMacro => "local-record-macro",
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
    RepeatedRecordScaffold,
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

    pub fn base_slice_symbol_count(&self) -> usize {
        self.segments
            .iter()
            .map(|segment| match segment {
                OverlaySegment::BaseSlice { len, .. } => *len,
                OverlaySegment::MacroRef { .. } => 0,
            })
            .sum()
    }

    pub fn macro_ref_symbol_count(&self) -> usize {
        self.macro_ref_count()
    }

    pub fn macro_definition_symbol_count(&self) -> usize {
        self.macros.iter().map(|entry| entry.token_ids.len()).sum()
    }

    pub fn overlay_symbol_count(&self) -> usize {
        self.base_slice_symbol_count()
            + self.macro_ref_symbol_count()
            + self.macro_definition_symbol_count()
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
    byte_start: usize,
    byte_end: usize,
    token_start: usize,
    token_end: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MacroCandidate {
    token_ids: Vec<u32>,
    count: usize,
    saved_mass: usize,
    kind: LocalMacroKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LineShapeGroup {
    shape_key: String,
    line_indices: Vec<usize>,
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
        RecursiveOverlayMode::LocalRecordMacro => {
            build_local_record_macro_overlay(text, canonical, config)
        }
    }
}

fn build_local_line_macro_overlay(
    text: &str,
    canonical: CanonicalTokenization,
    config: &RecursiveOverlayConfig,
) -> RecursiveOverlayDocument {
    let line_spans = collect_line_spans(text, &canonical.offsets);
    let macro_catalog = select_line_macros(&canonical.token_ids, &line_spans, config);
    if macro_catalog.is_empty() {
        return RecursiveOverlayDocument::passthrough(canonical);
    }

    let mut segments = Vec::new();
    let mut cursor = 0usize;
    let mut used_macro_ids = BTreeMap::<usize, usize>::new();
    for span in &line_spans {
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

    finalize_overlay_document(
        canonical,
        segments,
        used_macro_ids,
        macro_catalog.into_values().collect(),
    )
}

fn build_local_record_macro_overlay(
    text: &str,
    canonical: CanonicalTokenization,
    config: &RecursiveOverlayConfig,
) -> RecursiveOverlayDocument {
    let line_spans = collect_line_spans(text, &canonical.offsets);
    let groups = line_shape_groups(text, &line_spans);
    let macro_groups = select_record_macros(text, &canonical, &line_spans, &groups, config);
    if macro_groups.is_empty() {
        return RecursiveOverlayDocument::passthrough(canonical);
    }

    let mut segments = Vec::new();
    let mut cursor = 0usize;
    let mut used_macro_ids = BTreeMap::<usize, usize>::new();
    let line_to_shape = groups
        .iter()
        .flat_map(|group| {
            group
                .line_indices
                .iter()
                .map(move |line_index| (*line_index, group.shape_key.as_str()))
        })
        .collect::<BTreeMap<_, _>>();

    for (line_index, span) in line_spans.iter().enumerate() {
        if cursor < span.token_start {
            push_base_slice(&mut segments, cursor, span.token_start - cursor);
        }

        let line_tokens = &canonical.token_ids[span.token_start..span.token_end];
        if let Some(shape_key) = line_to_shape.get(&line_index) {
            if let Some(catalog) = macro_groups.get(*shape_key) {
                apply_macro_catalog_to_line(
                    &mut segments,
                    &mut used_macro_ids,
                    span.token_start,
                    line_tokens,
                    catalog,
                );
            } else {
                push_base_slice(
                    &mut segments,
                    span.token_start,
                    span.token_end - span.token_start,
                );
            }
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

    let all_macros = macro_groups.into_values().flatten().collect::<Vec<_>>();
    finalize_overlay_document(canonical, segments, used_macro_ids, all_macros)
}

fn finalize_overlay_document(
    canonical: CanonicalTokenization,
    segments: Vec<OverlaySegment>,
    used_macro_ids: BTreeMap<usize, usize>,
    all_macros: Vec<LocalMacro>,
) -> RecursiveOverlayDocument {
    let macros = all_macros
        .into_iter()
        .filter_map(|entry| {
            used_macro_ids
                .get(&entry.macro_id)
                .map(|use_count| LocalMacro {
                    use_count: *use_count,
                    ..entry
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
    line_spans: &[LineTokenSpan],
    config: &RecursiveOverlayConfig,
) -> BTreeMap<Vec<u32>, LocalMacro> {
    let mut candidates = BTreeMap::<Vec<u32>, usize>::new();
    for span in line_spans {
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
                kind: LocalMacroKind::RepeatedLine,
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
                kind: candidate.kind,
                token_ids: token_ids.clone(),
                use_count: candidate.count,
            };
            (token_ids, entry)
        })
        .collect()
}

fn collect_line_spans(text: &str, offsets: &[(usize, usize)]) -> Vec<LineTokenSpan> {
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
    for (line_start, line_end) in ranges {
        let token_start = cursor;
        while cursor < offsets.len() && offsets[cursor].0 < line_end {
            cursor += 1;
        }
        if cursor > token_start {
            spans.push(LineTokenSpan {
                byte_start: line_start,
                byte_end: line_end,
                token_start,
                token_end: cursor,
            });
        }
    }

    spans
}

fn line_shape_groups(text: &str, line_spans: &[LineTokenSpan]) -> Vec<LineShapeGroup> {
    let mut per_shape = BTreeMap::<String, Vec<usize>>::new();
    for (index, span) in line_spans.iter().enumerate() {
        let line = &text[span.byte_start..span.byte_end];
        per_shape
            .entry(normalized_record_shape(line))
            .or_default()
            .push(index);
    }

    per_shape
        .into_iter()
        .map(|(shape_key, line_indices)| LineShapeGroup {
            shape_key,
            line_indices,
        })
        .collect()
}

fn normalized_record_shape(line: &str) -> String {
    let mut shape = String::new();
    let mut last = '\0';
    for ch in line.chars() {
        let class = if ch.is_ascii_digit() {
            '#'
        } else if ch.is_ascii_alphabetic() || ch == '_' {
            'a'
        } else if ch.is_whitespace() {
            ' '
        } else {
            ch
        };
        if class != last || !matches!(class, '#' | 'a' | ' ') {
            shape.push(class);
            last = class;
        }
    }
    shape
}

fn select_record_macros(
    text: &str,
    canonical: &CanonicalTokenization,
    line_spans: &[LineTokenSpan],
    groups: &[LineShapeGroup],
    config: &RecursiveOverlayConfig,
) -> BTreeMap<String, Vec<LocalMacro>> {
    let max_span_tokens = config.max_span_tokens.min(24);
    if max_span_tokens < config.min_span_tokens {
        return BTreeMap::new();
    }

    let mut selected = BTreeMap::<String, Vec<LocalMacro>>::new();
    let mut next_macro_id = 0usize;
    for group in groups {
        if group.line_indices.len() < config.min_repeat_count {
            continue;
        }

        let mut candidates = BTreeMap::<Vec<u32>, (usize, usize)>::new();
        for line_index in &group.line_indices {
            let span = &line_spans[*line_index];
            let line_tokens = &canonical.token_ids[span.token_start..span.token_end];
            let mut seen_in_line = BTreeMap::<Vec<u32>, usize>::new();

            for start in 0..line_tokens.len() {
                let max_len = (line_tokens.len() - start).min(max_span_tokens);
                for len in config.min_span_tokens..=max_len {
                    let slice_start = span.token_start + start;
                    let slice_end = slice_start + len;
                    if !span_has_structural_marker(text, canonical, slice_start, slice_end) {
                        continue;
                    }
                    let slice = line_tokens[start..start + len].to_vec();
                    *seen_in_line.entry(slice).or_default() += 1;
                }
            }

            for (slice, count_in_line) in seen_in_line {
                let entry = candidates.entry(slice).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += count_in_line;
            }
        }

        let mut group_selected = candidates
            .into_iter()
            .filter_map(|(token_ids, (line_hits, occurrences))| {
                if line_hits < config.min_repeat_count {
                    return None;
                }
                let saved_mass = token_ids
                    .len()
                    .saturating_mul(occurrences.saturating_sub(1));
                if saved_mass < config.min_repeated_token_mass {
                    return None;
                }
                Some(MacroCandidate {
                    token_ids,
                    count: occurrences,
                    saved_mass,
                    kind: LocalMacroKind::RepeatedRecordScaffold,
                })
            })
            .collect::<Vec<_>>();

        group_selected.sort_by(|left, right| {
            right
                .saved_mass
                .cmp(&left.saved_mass)
                .then_with(|| right.token_ids.len().cmp(&left.token_ids.len()))
                .then_with(|| left.token_ids.cmp(&right.token_ids))
        });

        if group_selected.is_empty() {
            continue;
        }

        let entries = group_selected
            .into_iter()
            .map(|candidate| {
                let macro_id = next_macro_id;
                next_macro_id += 1;
                LocalMacro {
                    macro_id,
                    kind: candidate.kind,
                    token_ids: candidate.token_ids,
                    use_count: candidate.count,
                }
            })
            .collect::<Vec<_>>();
        selected.insert(group.shape_key.clone(), entries);
    }

    selected
}

fn apply_macro_catalog_to_line(
    segments: &mut Vec<OverlaySegment>,
    used_macro_ids: &mut BTreeMap<usize, usize>,
    base_start: usize,
    line_tokens: &[u32],
    catalog: &[LocalMacro],
) {
    let mut cursor = 0usize;
    while cursor < line_tokens.len() {
        let next_match = find_next_macro_match(line_tokens, cursor, catalog);
        let Some((match_start, macro_entry)) = next_match else {
            push_base_slice(segments, base_start + cursor, line_tokens.len() - cursor);
            break;
        };

        if match_start > cursor {
            push_base_slice(segments, base_start + cursor, match_start - cursor);
        }

        segments.push(OverlaySegment::MacroRef {
            macro_id: macro_entry.macro_id,
            span_len: macro_entry.token_ids.len(),
        });
        *used_macro_ids.entry(macro_entry.macro_id).or_default() += 1;
        cursor = match_start + macro_entry.token_ids.len();
    }
}

fn find_next_macro_match<'a>(
    line_tokens: &[u32],
    cursor: usize,
    catalog: &'a [LocalMacro],
) -> Option<(usize, &'a LocalMacro)> {
    let mut best: Option<(usize, &LocalMacro)> = None;
    for start in cursor..line_tokens.len() {
        for macro_entry in catalog {
            let len = macro_entry.token_ids.len();
            if start + len > line_tokens.len() {
                continue;
            }
            if line_tokens[start..start + len] == macro_entry.token_ids[..] {
                match best {
                    Some((best_start, best_macro))
                        if start > best_start
                            || (start == best_start && len <= best_macro.token_ids.len()) => {}
                    _ => best = Some((start, macro_entry)),
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    best
}

fn span_has_structural_marker(
    text: &str,
    canonical: &CanonicalTokenization,
    token_start: usize,
    token_end: usize,
) -> bool {
    let Some((byte_start, byte_end)) = span_byte_range(canonical, token_start, token_end) else {
        return false;
    };
    text[byte_start..byte_end].chars().any(|ch| {
        matches!(
            ch,
            '{' | '}' | '[' | ']' | ':' | '=' | ',' | '"' | '\'' | '/'
        )
    })
}

fn span_byte_range(
    canonical: &CanonicalTokenization,
    token_start: usize,
    token_end: usize,
) -> Option<(usize, usize)> {
    let start = canonical.offsets.get(token_start)?.0;
    let end = canonical.offsets.get(token_end.checked_sub(1)?)?.1;
    Some((start, end))
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
