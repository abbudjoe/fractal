// Naming Convention for tokenizer primitives:
// [base]_[lever-description]_v[version]
// Examples: p1_fractal_hybrid_v1, b1_fractal_gated_dyn-residual-norm_v1
use burn::{
    module::Param,
    nn::Linear,
    tensor::{backend::Backend, Element, Tensor, TensorData},
};
use fractal_core::{
    error::FractalError,
    rule_trait::{ApplyContext, FractalRule},
    state::{FractalState, StateLayout},
};
use std::collections::{BTreeMap, BTreeSet};

use crate::{
    faceoff::{scan_lexemes, LexemeSpan},
    B1FractalGated, B3FractalHierarchical, B4Universal, FaceoffLexemeKind, P1FractalHybrid,
    P2Mandelbrot,
};

const DEFAULT_LEVELS: usize = 3;
const TRACKER_REMINDER: &str =
    "Reminder: update docs/tokenizer-tracker.md with the latest tokenizer results.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitPolicy {
    Balanced,
    BoundaryAware,
    SyntaxAware,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TokenizerSubstrateMode {
    #[default]
    RawBytes,
    LexicalAtoms,
}

#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub dim: usize,
    pub levels: usize,
    pub max_depth: usize,
    pub seed: u64,
    pub split_policy: SplitPolicy,
    pub substrate_mode: TokenizerSubstrateMode,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            levels: DEFAULT_LEVELS,
            max_depth: 6,
            seed: 42,
            split_policy: SplitPolicy::Balanced,
            substrate_mode: TokenizerSubstrateMode::RawBytes,
        }
    }
}

const STATE_SIGNATURE_PREFIX_WIDTH: usize = 6;
const STATE_SIGNATURE_SUFFIX_WIDTH: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateSignature {
    pub state_bin: u16,
    pub norm_bin: u8,
    pub mean_abs_bin: u8,
    pub prefix_bins: [i8; STATE_SIGNATURE_PREFIX_WIDTH],
    pub suffix_bins: [i8; STATE_SIGNATURE_SUFFIX_WIDTH],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRecord {
    pub depth: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub token: String,
    pub state_signature: StateSignature,
}

#[derive(Clone, Debug)]
pub struct PrimitiveRunSummary {
    pub primitive: &'static str,
    pub produced: usize,
    pub tokens: Vec<TokenRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotifReusePolicy {
    Off,
    StateNormSimilarityV2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerPrimitiveName(&'static str);

impl TokenizerPrimitiveName {
    pub fn new(name: &'static str) -> Result<Self, FractalError> {
        validate_tokenizer_primitive_name(name)?;
        Ok(Self(name))
    }

    pub fn as_str(self) -> &'static str {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrimitiveFactory<B: Backend> {
    pub name: &'static str,
    pub motif_reuse: MotifReusePolicy,
    pub build: fn(TokenizerConfig, &B::Device) -> Box<dyn FractalRule<B>>,
}

impl<B: Backend> PrimitiveFactory<B> {
    pub fn new(
        name: &'static str,
        motif_reuse: MotifReusePolicy,
        build: fn(TokenizerConfig, &B::Device) -> Box<dyn FractalRule<B>>,
    ) -> Self {
        Self {
            name,
            motif_reuse,
            build,
        }
    }

    pub fn try_new(
        name: &'static str,
        motif_reuse: MotifReusePolicy,
        build: fn(TokenizerConfig, &B::Device) -> Box<dyn FractalRule<B>>,
    ) -> Result<Self, FractalError> {
        let validated_name = TokenizerPrimitiveName::new(name)?;
        Ok(Self::new(validated_name.as_str(), motif_reuse, build))
    }

    pub fn validated_name(&self) -> Result<TokenizerPrimitiveName, FractalError> {
        TokenizerPrimitiveName::new(self.name)
    }
}

pub struct RecursiveTokenizer {
    config: TokenizerConfig,
}

#[derive(Default)]
struct MotifRegistry {
    assigned_by_depth: BTreeMap<usize, BTreeSet<String>>,
    seen: Vec<SeenMotif>,
}

#[derive(Clone, Copy)]
struct Segment<'a> {
    bytes: &'a [u8],
    start: usize,
    depth: usize,
}

#[derive(Clone, Copy)]
struct AtomSegment {
    start_index: usize,
    end_index: usize,
    depth: usize,
}

#[derive(Clone)]
struct SeenMotif {
    depth: usize,
    digest: String,
    norm: f32,
    signature: Vec<f32>,
}

struct TokenSummary {
    token: String,
    state_signature: StateSignature,
}

struct TokenizeContext<'a, B: Backend> {
    rule: &'a dyn FractalRule<B>,
    tokens: &'a mut Vec<TokenRecord>,
    motifs: &'a mut MotifRegistry,
    motif_reuse: MotifReusePolicy,
    device: &'a B::Device,
}

#[derive(Clone, Copy)]
struct AtomTokenizeInput<'a> {
    atoms: &'a [LexemeSpan],
    text: &'a str,
}

impl RecursiveTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self { config }
    }

    pub fn tokenize<B: Backend>(
        &self,
        rule: &dyn FractalRule<B>,
        text: &str,
        device: &B::Device,
    ) -> Result<Vec<TokenRecord>, FractalError> {
        self.tokenize_with_policy(rule, text, device, MotifReusePolicy::Off)
    }

    fn tokenize_with_policy<B: Backend>(
        &self,
        rule: &dyn FractalRule<B>,
        text: &str,
        device: &B::Device,
        motif_reuse: MotifReusePolicy,
    ) -> Result<Vec<TokenRecord>, FractalError> {
        let state = FractalState::zeros(rule.state_layout(), 1, rule.hidden_dim(), device)?;
        let mut tokens = Vec::new();
        let mut motifs = MotifRegistry::default();
        match self.config.substrate_mode {
            TokenizerSubstrateMode::RawBytes => {
                let root = Segment {
                    bytes: text.as_bytes(),
                    start: 0,
                    depth: 0,
                };
                let mut context = TokenizeContext {
                    rule,
                    tokens: &mut tokens,
                    motifs: &mut motifs,
                    motif_reuse,
                    device,
                };
                self.tokenize_segment(state, root, &mut context)?;
            }
            TokenizerSubstrateMode::LexicalAtoms => {
                let atoms = scan_lexemes(text, 0);
                if atoms.is_empty() {
                    return Ok(tokens);
                }
                let root = AtomSegment {
                    start_index: 0,
                    end_index: atoms.len(),
                    depth: 0,
                };
                let atom_input = AtomTokenizeInput {
                    atoms: &atoms,
                    text,
                };
                let mut context = TokenizeContext {
                    rule,
                    tokens: &mut tokens,
                    motifs: &mut motifs,
                    motif_reuse,
                    device,
                };
                self.tokenize_atom_segment(state, atom_input, root, &mut context)?;
            }
        }
        Ok(tokens)
    }

    pub fn run_factory<B: Backend>(
        &self,
        text: &str,
        device: &B::Device,
        factory: PrimitiveFactory<B>,
    ) -> Result<PrimitiveRunSummary, FractalError> {
        let _ = factory.validated_name()?;
        let rule = (factory.build)(self.config, device);
        let tokens = self.tokenize_with_policy(rule.as_ref(), text, device, factory.motif_reuse)?;
        Ok(PrimitiveRunSummary {
            primitive: factory.name,
            produced: tokens.len(),
            tokens,
        })
    }

    fn tokenize_segment<B: Backend>(
        &self,
        state: FractalState<B>,
        segment: Segment<'_>,
        context: &mut TokenizeContext<'_, B>,
    ) -> Result<(), FractalError> {
        if segment.bytes.is_empty() {
            return Ok(());
        }

        let features = segment_features::<B>(segment.bytes, self.config.dim, context.device);
        let next_state = context.rule.apply(
            &state,
            &features,
            ApplyContext {
                depth: segment.depth,
                max_depth: self.config.max_depth,
            },
        )?;
        let summary = summarize_readout(
            &next_state.readout(),
            segment.bytes,
            segment.depth,
            context.motifs,
            context.motif_reuse,
        )?;
        let end = segment.start + segment.bytes.len();
        context.tokens.push(TokenRecord {
            depth: segment.depth,
            start: segment.start,
            end,
            text: String::from_utf8_lossy(segment.bytes).into_owned(),
            token: summary.token,
            state_signature: summary.state_signature,
        });

        if segment.depth + 1 >= self.config.max_depth || segment.bytes.len() <= 1 {
            return Ok(());
        }

        if let Some(split) = split_point(segment.bytes, self.config.split_policy) {
            let (left, right) = segment.bytes.split_at(split);
            self.tokenize_segment(
                next_state.clone(),
                Segment {
                    bytes: left,
                    start: segment.start,
                    depth: segment.depth + 1,
                },
                context,
            )?;
            self.tokenize_segment(
                next_state,
                Segment {
                    bytes: right,
                    start: segment.start + split,
                    depth: segment.depth + 1,
                },
                context,
            )?;
        }

        Ok(())
    }

    fn tokenize_atom_segment<B: Backend>(
        &self,
        state: FractalState<B>,
        input: AtomTokenizeInput<'_>,
        segment: AtomSegment,
        context: &mut TokenizeContext<'_, B>,
    ) -> Result<(), FractalError> {
        if segment.start_index >= segment.end_index || segment.end_index > input.atoms.len() {
            return Ok(());
        }

        let atom_slice = &input.atoms[segment.start_index..segment.end_index];
        let byte_start = atom_slice.first().map(|atom| atom.start).unwrap_or(0);
        let byte_end = atom_slice.last().map(|atom| atom.end).unwrap_or(byte_start);
        if byte_end > input.text.len() || byte_start > byte_end {
            return Err(FractalError::InvalidState(format!(
                "atom segment span {}..{} is out of bounds for input length {}",
                byte_start,
                byte_end,
                input.text.len()
            )));
        }

        let bytes = &input.text.as_bytes()[byte_start..byte_end];
        let features = atom_segment_features::<B>(atom_slice, self.config.dim, context.device);
        let next_state = context.rule.apply(
            &state,
            &features,
            ApplyContext {
                depth: segment.depth,
                max_depth: self.config.max_depth,
            },
        )?;
        let summary = summarize_readout(
            &next_state.readout(),
            bytes,
            segment.depth,
            context.motifs,
            context.motif_reuse,
        )?;
        context.tokens.push(TokenRecord {
            depth: segment.depth,
            start: byte_start,
            end: byte_end,
            text: input.text[byte_start..byte_end].to_string(),
            token: summary.token,
            state_signature: summary.state_signature,
        });

        if segment.depth + 1 >= self.config.max_depth || atom_slice.len() <= 1 {
            return Ok(());
        }

        if let Some(split) = atom_split_point(atom_slice, self.config.split_policy) {
            let absolute_split = segment.start_index + split;
            self.tokenize_atom_segment(
                next_state.clone(),
                input,
                AtomSegment {
                    start_index: segment.start_index,
                    end_index: absolute_split,
                    depth: segment.depth + 1,
                },
                context,
            )?;
            self.tokenize_atom_segment(
                next_state,
                input,
                AtomSegment {
                    start_index: absolute_split,
                    end_index: segment.end_index,
                    depth: segment.depth + 1,
                },
                context,
            )?;
        }

        Ok(())
    }
}

pub fn revived_primitive_factories<B: Backend>() -> [PrimitiveFactory<B>; 5] {
    [
        PrimitiveFactory::new(
            "b1_fractal_gated_v1",
            MotifReusePolicy::Off,
            seeded_b1_fractal_gated::<B>,
        ),
        PrimitiveFactory::new(
            "p1_fractal_hybrid_v1",
            MotifReusePolicy::Off,
            seeded_p1_fractal_hybrid::<B>,
        ),
        PrimitiveFactory::new(
            "p2_mandelbrot_v1",
            MotifReusePolicy::Off,
            seeded_p2_mandelbrot::<B>,
        ),
        PrimitiveFactory::new(
            "b3_fractal_hierarchical_v1",
            MotifReusePolicy::Off,
            seeded_b3_fractal_hierarchical::<B>,
        ),
        PrimitiveFactory::new(
            "b4_universal_v1",
            MotifReusePolicy::Off,
            seeded_b4_universal::<B>,
        ),
    ]
}

pub fn try_p1_dynamic_lever_factory<B: Backend>() -> Result<PrimitiveFactory<B>, FractalError> {
    PrimitiveFactory::try_new(
        "p1_fractal_hybrid_dyn-state-norm_v2",
        MotifReusePolicy::StateNormSimilarityV2,
        seeded_p1_fractal_hybrid_dynamic::<B>,
    )
}

pub fn p1_dynamic_lever_factory<B: Backend>() -> PrimitiveFactory<B> {
    PrimitiveFactory::new(
        "p1_fractal_hybrid_dyn-state-norm_v2",
        MotifReusePolicy::StateNormSimilarityV2,
        seeded_p1_fractal_hybrid_dynamic::<B>,
    )
}

pub fn validate_tokenizer_primitive_name(name: &str) -> Result<(), FractalError> {
    let (prefix, version) = name.rsplit_once("_v").ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "tokenizer primitive `{name}` must end with `_v[version]`"
        ))
    })?;

    if version.is_empty() || !version.chars().all(|ch| ch.is_ascii_digit()) {
        return Err(FractalError::InvalidConfig(format!(
            "tokenizer primitive `{name}` must use a numeric version suffix"
        )));
    }

    let (base, lever) = prefix.split_once('_').ok_or_else(|| {
        FractalError::InvalidConfig(format!(
            "tokenizer primitive `{name}` must follow `[base]_[lever-description]_v[version]`"
        ))
    })?;

    if !is_valid_base_segment(base) {
        return Err(FractalError::InvalidConfig(format!(
            "tokenizer primitive `{name}` has an invalid base segment `{base}`"
        )));
    }

    if !is_valid_lever_segment(lever) {
        return Err(FractalError::InvalidConfig(format!(
            "tokenizer primitive `{name}` has an invalid lever segment `{lever}`"
        )));
    }

    Ok(())
}

pub fn tokenizer_tracker_reminder() -> &'static str {
    TRACKER_REMINDER
}

fn seeded_b1_fractal_gated<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = B1FractalGated::new(config.dim, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        0,
        device,
    );
    seed_linear(
        &mut rule.c_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        1,
        device,
    );
    Box::new(rule)
}

fn seeded_p1_fractal_hybrid<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = P1FractalHybrid::new_with_dynamic_lever(config.dim, false, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim,
        config.seed,
        2,
        device,
    );
    seed_linear(
        &mut rule.w_h,
        config.dim,
        config.dim,
        config.seed,
        3,
        device,
    );
    seed_linear(&mut rule.u, config.dim, config.dim, config.seed, 4, device);
    Box::new(rule)
}

fn seeded_p1_fractal_hybrid_dynamic<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = P1FractalHybrid::new_with_dynamic_lever(config.dim, true, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim,
        config.seed,
        2,
        device,
    );
    seed_linear(
        &mut rule.w_h,
        config.dim,
        config.dim,
        config.seed,
        3,
        device,
    );
    seed_linear(&mut rule.u, config.dim, config.dim, config.seed, 4, device);
    Box::new(rule)
}

fn seeded_p2_mandelbrot<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = P2Mandelbrot::new(config.dim, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        5,
        device,
    );
    seed_linear(
        &mut rule.c_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        6,
        device,
    );
    Box::new(rule)
}

fn seeded_b3_fractal_hierarchical<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = B3FractalHierarchical::new(config.dim, config.levels, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        7,
        device,
    );
    seed_linear(
        &mut rule.c_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        8,
        device,
    );
    seed_linear(
        &mut rule.gamma_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        9,
        device,
    );
    seed_linear(
        &mut rule.compressor,
        config.dim * 2,
        config.dim * 2,
        config.seed,
        10,
        device,
    );
    Box::new(rule)
}

fn seeded_b4_universal<B: Backend>(
    config: TokenizerConfig,
    device: &B::Device,
) -> Box<dyn FractalRule<B>> {
    let mut rule = B4Universal::new(config.dim, config.levels, device);
    seed_linear(
        &mut rule.g_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        11,
        device,
    );
    seed_linear(
        &mut rule.c_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        12,
        device,
    );
    seed_linear(
        &mut rule.gamma_proj,
        config.dim,
        config.dim * 2,
        config.seed,
        13,
        device,
    );
    seed_linear(
        &mut rule.compressor,
        config.dim * 2,
        config.dim * 2,
        config.seed,
        14,
        device,
    );
    Box::new(rule)
}

fn seed_linear<B: Backend>(
    linear: &mut Linear<B>,
    d_input: usize,
    d_output: usize,
    seed: u64,
    stream: u64,
    device: &B::Device,
) {
    let mut state = mix_seed(seed, stream);
    let weights = (0..d_input * d_output)
        .map(|_| next_weight(&mut state))
        .collect::<Vec<_>>();
    linear.weight = Param::from_data(TensorData::new(weights, [d_input, d_output]), device);

    if linear.bias.is_some() {
        let bias = (0..d_output)
            .map(|_| next_weight(&mut state))
            .collect::<Vec<_>>();
        linear.bias = Some(Param::from_data(TensorData::new(bias, [d_output]), device));
    }
}

fn mix_seed(seed: u64, stream: u64) -> u64 {
    let mut state = seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^ (state >> 31)
}

fn next_weight(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let unit = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
    ((unit as f32) * 2.0 - 1.0) * 0.125
}

fn is_valid_base_segment(base: &str) -> bool {
    !base.is_empty()
        && base
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
}

fn is_valid_lever_segment(lever: &str) -> bool {
    !lever.is_empty()
        && lever
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
}

fn split_point(bytes: &[u8], policy: SplitPolicy) -> Option<usize> {
    if bytes.len() <= 1 {
        return None;
    }

    let fallback = balanced_split_point(bytes);
    match policy {
        SplitPolicy::Balanced => fallback,
        SplitPolicy::BoundaryAware => boundary_aware_split_point(bytes).or(fallback),
        SplitPolicy::SyntaxAware => syntax_aware_byte_split_point(bytes)
            .or_else(|| boundary_aware_split_point(bytes))
            .or(fallback),
    }
}

fn atom_split_point(atoms: &[LexemeSpan], policy: SplitPolicy) -> Option<usize> {
    if atoms.len() <= 1 {
        return None;
    }

    let fallback = Some((atoms.len() / 2).max(1));
    match policy {
        SplitPolicy::Balanced => fallback,
        SplitPolicy::BoundaryAware => atom_boundary_aware_split_point(atoms).or(fallback),
        SplitPolicy::SyntaxAware => syntax_aware_atom_split_point(atoms)
            .or_else(|| atom_boundary_aware_split_point(atoms))
            .or(fallback),
    }
}

fn balanced_split_point(bytes: &[u8]) -> Option<usize> {
    let mid = bytes.len() / 2;
    let mut best = None;
    for index in 1..bytes.len() {
        if bytes[index - 1].is_ascii_whitespace() || bytes[index].is_ascii_whitespace() {
            let distance = mid.abs_diff(index);
            if best.is_none_or(|(_, best_distance)| distance < best_distance) {
                best = Some((index, distance));
            }
        }
    }

    Some(best.map(|(index, _)| index).unwrap_or(mid.max(1)))
}

fn atom_boundary_aware_split_point(atoms: &[LexemeSpan]) -> Option<usize> {
    if atoms.len() <= 1 {
        return None;
    }

    let midpoint = atoms.len() / 2;
    let mut best: Option<(u8, usize, usize)> = None;

    for index in 1..atoms.len() {
        let boundary = classify_atom_boundary(&atoms[index - 1], &atoms[index]);
        let distance = midpoint.abs_diff(index);
        let candidate = (boundary, distance, index);
        if best.is_none_or(|current| candidate < current) {
            best = Some(candidate);
        }
    }

    best.map(|(_, _, index)| index)
}

fn syntax_aware_atom_split_point(atoms: &[LexemeSpan]) -> Option<usize> {
    if atoms.len() <= 1 {
        return None;
    }

    let midpoint = atoms.len() / 2;
    let mut best: Option<(u8, usize, usize)> = None;

    for index in 1..atoms.len() {
        let boundary = classify_syntax_atom_boundary(atoms, index);
        let distance = midpoint.abs_diff(index);
        let candidate = (boundary, distance, index);
        if best.is_none_or(|current| candidate < current) {
            best = Some(candidate);
        }
    }

    best.map(|(_, _, index)| index)
}

fn boundary_aware_split_point(bytes: &[u8]) -> Option<usize> {
    let text = std::str::from_utf8(bytes).ok()?;
    if text.len() <= 1 {
        return None;
    }

    let midpoint = text.len() / 2;
    let min_child_len = 1usize;
    let mut best: Option<(u8, usize, usize)> = None;

    for (index, _) in text.char_indices().skip(1) {
        if index <= min_child_len || text.len().saturating_sub(index) <= min_child_len {
            continue;
        }
        let boundary = classify_boundary(text, index)?;
        let distance = midpoint.abs_diff(index);
        let candidate = (boundary, distance, index);
        if best.is_none_or(|current| candidate < current) {
            best = Some(candidate);
        }
    }

    best.map(|(_, _, index)| index)
}

fn syntax_aware_byte_split_point(bytes: &[u8]) -> Option<usize> {
    boundary_aware_split_point(bytes)
}

fn classify_atom_boundary(left: &LexemeSpan, right: &LexemeSpan) -> u8 {
    if left.kind == FaceoffLexemeKind::NewlineIndent
        && right.kind == FaceoffLexemeKind::NewlineIndent
    {
        return 0;
    }
    if left.kind == FaceoffLexemeKind::NewlineIndent
        || right.kind == FaceoffLexemeKind::NewlineIndent
    {
        return 1;
    }
    if left.kind == FaceoffLexemeKind::Punctuation || right.kind == FaceoffLexemeKind::Punctuation {
        return 2;
    }
    if left.kind == FaceoffLexemeKind::Whitespace || right.kind == FaceoffLexemeKind::Whitespace {
        return 3;
    }
    4
}

fn classify_syntax_atom_boundary(atoms: &[LexemeSpan], index: usize) -> u8 {
    let left = &atoms[index - 1];
    let right = &atoms[index];

    if is_blank_line_boundary(left, right) {
        return 0;
    }
    if is_line_start_declaration_boundary(atoms, index)
        || is_pre_line_start_declaration_boundary(atoms, index)
    {
        return 1;
    }
    if is_line_start_markdown_boundary(atoms, index)
        || is_pre_line_start_markdown_boundary(atoms, index)
    {
        return 2;
    }
    if is_statement_or_block_boundary(left, right) {
        return 3;
    }
    if left.kind == FaceoffLexemeKind::Punctuation || right.kind == FaceoffLexemeKind::Punctuation {
        return 4;
    }
    if left.kind == FaceoffLexemeKind::Whitespace || right.kind == FaceoffLexemeKind::Whitespace {
        return 5;
    }

    6
}

fn classify_boundary(text: &str, index: usize) -> Option<u8> {
    let prev = text[..index].chars().next_back()?;
    let next = text[index..].chars().next()?;

    if text[..index].ends_with("\n\n") || text[index..].starts_with("\n\n") {
        return Some(0);
    }
    if prev == '\n' || next == '\n' {
        return Some(1);
    }
    if is_structural_punctuation(prev) || is_structural_punctuation(next) {
        return Some(2);
    }
    if prev.is_whitespace() || next.is_whitespace() {
        return Some(3);
    }

    None
}

fn is_blank_line_boundary(left: &LexemeSpan, right: &LexemeSpan) -> bool {
    left.kind == FaceoffLexemeKind::NewlineIndent && right.kind == FaceoffLexemeKind::NewlineIndent
}

fn is_line_start_declaration_boundary(atoms: &[LexemeSpan], index: usize) -> bool {
    if !is_line_start(atoms, index) {
        return false;
    }
    lexeme_text(&atoms[index]).is_some_and(is_declaration_keyword)
}

fn is_pre_line_start_declaration_boundary(atoms: &[LexemeSpan], index: usize) -> bool {
    atoms[index].kind == FaceoffLexemeKind::NewlineIndent
        && atoms
            .get(index + 1)
            .and_then(lexeme_text)
            .is_some_and(is_declaration_keyword)
}

fn is_line_start_markdown_boundary(atoms: &[LexemeSpan], index: usize) -> bool {
    if !is_line_start(atoms, index) {
        return false;
    }

    let current = &atoms[index];
    if lexeme_text(current).is_some_and(is_markdown_heading_marker) {
        return true;
    }
    if lexeme_text(current).is_some_and(is_list_marker) {
        return true;
    }

    current.kind == FaceoffLexemeKind::Number
        && atoms
            .get(index + 1)
            .and_then(lexeme_text)
            .is_some_and(is_enumerated_list_suffix)
}

fn is_pre_line_start_markdown_boundary(atoms: &[LexemeSpan], index: usize) -> bool {
    if atoms[index].kind != FaceoffLexemeKind::NewlineIndent {
        return false;
    }

    let Some(next) = atoms.get(index + 1) else {
        return false;
    };
    if lexeme_text(next).is_some_and(is_markdown_heading_marker) {
        return true;
    }
    if lexeme_text(next).is_some_and(is_list_marker) {
        return true;
    }

    next.kind == FaceoffLexemeKind::Number
        && atoms
            .get(index + 2)
            .and_then(lexeme_text)
            .is_some_and(is_enumerated_list_suffix)
}

fn is_statement_or_block_boundary(left: &LexemeSpan, right: &LexemeSpan) -> bool {
    if right.kind == FaceoffLexemeKind::NewlineIndent
        && lexeme_text(left).is_some_and(is_clause_terminator)
    {
        return true;
    }

    lexeme_text(left).is_some_and(|text| matches!(text, "}" | "{"))
        || lexeme_text(right).is_some_and(|text| matches!(text, "}" | "{"))
}

fn is_line_start(atoms: &[LexemeSpan], index: usize) -> bool {
    index == 0 || atoms[index - 1].kind == FaceoffLexemeKind::NewlineIndent
}

fn lexeme_text(span: &LexemeSpan) -> Option<&str> {
    std::str::from_utf8(&span.bytes).ok()
}

fn is_declaration_keyword(text: &str) -> bool {
    matches!(
        text,
        "fn" | "struct"
            | "enum"
            | "impl"
            | "trait"
            | "use"
            | "pub"
            | "const"
            | "let"
            | "match"
            | "if"
            | "for"
            | "while"
            | "return"
            | "class"
            | "actor"
            | "protocol"
            | "extension"
            | "func"
            | "var"
            | "import"
    )
}

fn is_markdown_heading_marker(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte == b'#')
}

fn is_list_marker(text: &str) -> bool {
    matches!(text, "-" | "*")
}

fn is_enumerated_list_suffix(text: &str) -> bool {
    matches!(text, "." | ")")
}

fn is_clause_terminator(text: &str) -> bool {
    matches!(text, "}" | ";" | ":" | "." | "?" | "!")
}

fn is_structural_punctuation(value: char) -> bool {
    matches!(
        value,
        ';' | ':' | ',' | '.' | '{' | '}' | '(' | ')' | '[' | ']' | '#'
    )
}

fn segment_features<B: Backend>(bytes: &[u8], dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut features = vec![0.0f32; dim];
    if bytes.is_empty() {
        return Tensor::zeros([1, dim], device);
    }

    let mut whitespace = 0.0f32;
    let mut punctuation = 0.0f32;
    let mut uppercase = 0.0f32;

    for (index, byte) in bytes.iter().copied().enumerate() {
        let centered = byte as f32 / 127.5 - 1.0;
        let bucket = index % dim;
        let mirror = dim - 1 - bucket;
        features[bucket] += centered;
        features[mirror] += centered * 0.5;
        if byte.is_ascii_whitespace() {
            whitespace += 1.0;
        }
        if byte.is_ascii_punctuation() {
            punctuation += 1.0;
        }
        if byte.is_ascii_uppercase() {
            uppercase += 1.0;
        }
    }

    let len = bytes.len() as f32;
    for value in &mut features {
        *value /= len;
    }

    if dim >= 4 {
        features[dim - 4] = len / dim.max(1) as f32;
        features[dim - 3] = whitespace / len;
        features[dim - 2] = punctuation / len;
        features[dim - 1] = uppercase / len;
    }

    Tensor::from_data(TensorData::new(features, [1, dim]), device)
}

fn atom_segment_features<B: Backend>(
    atoms: &[LexemeSpan],
    dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut features = vec![0.0f32; dim];
    if atoms.is_empty() {
        return Tensor::zeros([1, dim], device);
    }

    let atom_count = atoms.len() as f32;
    let byte_len = atoms
        .iter()
        .map(|atom| atom.end.saturating_sub(atom.start))
        .sum::<usize>() as f32;
    let transition_count = atoms.len().saturating_sub(1) as f32;

    for atom in atoms {
        let kind_bucket = atom.kind.stable_index() as usize;
        if kind_bucket < dim {
            features[kind_bucket] += 1.0 / atom_count;
        }
    }

    if dim > 8 {
        let transition_buckets = dim - 8;
        for window in atoms.windows(2) {
            let pair_index =
                (window[0].kind.stable_index() * 16 + window[1].kind.stable_index()) as usize;
            let bucket = 7 + (pair_index % transition_buckets);
            features[bucket] += 1.0 / transition_count.max(1.0);
        }
    }

    if dim >= 4 {
        features[dim - 4] = atom_count / dim.max(1) as f32;
        features[dim - 3] = byte_len / dim.max(1) as f32;
        features[dim - 2] = atoms
            .iter()
            .filter(|atom| atom.kind == FaceoffLexemeKind::NewlineIndent)
            .count() as f32
            / atom_count;
        features[dim - 1] = atoms
            .iter()
            .filter(|atom| {
                matches!(
                    atom.kind,
                    FaceoffLexemeKind::Punctuation | FaceoffLexemeKind::SymbolRun
                )
            })
            .count() as f32
            / atom_count;
    }

    Tensor::from_data(TensorData::new(features, [1, dim]), device)
}

fn summarize_readout<B: Backend>(
    readout: &Tensor<B, 2>,
    bytes: &[u8],
    depth: usize,
    motifs: &mut MotifRegistry,
    motif_reuse: MotifReusePolicy,
) -> Result<TokenSummary, FractalError> {
    let values = tensor_data_to_vec::<f32>(readout.clone().into_data(), "token readout")?;
    let state_signature = StateSignature::from_values(&values);
    let prefix = values.iter().take(8).copied().collect::<Vec<_>>();
    let mut digest = 1469598103934665603u64;
    for value in prefix {
        let quantized = (value * 1000.0).round() as i64;
        digest ^= quantized as u64;
        digest = digest.wrapping_mul(1099511628211);
    }
    let digest = format!("{digest:016x}");
    let motif = motifs.resolve(depth, digest, &values, normalized_l2(&values), motif_reuse);
    Ok(TokenSummary {
        token: format!(
            "d{depth}-n{}-q{}-{motif}",
            bytes.len(),
            state_signature.state_bin
        ),
        state_signature,
    })
}

fn tensor_data_to_vec<E: Element>(
    data: TensorData,
    label: &'static str,
) -> Result<Vec<E>, FractalError> {
    data.to_vec::<E>()
        .map_err(|err| FractalError::InvalidState(format!("failed to extract {label}: {err:?}")))
}

#[allow(dead_code)]
fn _layout_width(layout: StateLayout, hidden_dim: usize) -> usize {
    layout.readout_width(hidden_dim)
}

impl MotifRegistry {
    fn resolve(
        &mut self,
        depth: usize,
        digest: String,
        signature: &[f32],
        rolling_norm: f32,
        motif_reuse: MotifReusePolicy,
    ) -> String {
        let resolved = match motif_reuse {
            MotifReusePolicy::Off => digest.clone(),
            MotifReusePolicy::StateNormSimilarityV2 => {
                if self.reuse_gate_open(rolling_norm) {
                    let candidates = self
                        .seen
                        .iter()
                        .filter(|entry| entry.depth != depth)
                        .filter(|entry| !self.digest_used_at_depth(depth, &entry.digest))
                        .map(|entry| (signature_distance(signature, &entry.signature), entry))
                        .collect::<Vec<_>>();
                    let nearest = candidates
                        .iter()
                        .min_by(|left, right| left.0.total_cmp(&right.0));
                    let adaptive_threshold = dynamic_similarity_threshold_v2(
                        signature,
                        depth,
                        rolling_norm,
                        &candidates,
                    );

                    nearest
                        .filter(|(distance, _)| *distance <= adaptive_threshold)
                        .map(|(_, entry)| entry.digest.clone())
                        .unwrap_or_else(|| digest.clone())
                } else {
                    digest.clone()
                }
            }
        };

        self.assigned_by_depth
            .entry(depth)
            .or_default()
            .insert(resolved.clone());
        self.seen.push(SeenMotif {
            depth,
            digest,
            norm: rolling_norm,
            signature: signature.to_vec(),
        });
        resolved
    }

    fn digest_used_at_depth(&self, depth: usize, digest: &str) -> bool {
        self.assigned_by_depth
            .get(&depth)
            .is_some_and(|digests| digests.contains(digest))
    }

    fn reuse_gate_open(&self, rolling_norm: f32) -> bool {
        self.mean_norm()
            .is_none_or(|mean_norm| rolling_norm >= mean_norm)
    }

    fn mean_norm(&self) -> Option<f32> {
        (!self.seen.is_empty())
            .then(|| self.seen.iter().map(|entry| entry.norm).sum::<f32>() / self.seen.len() as f32)
    }
}

fn dynamic_similarity_threshold_v2(
    signature: &[f32],
    depth: usize,
    rolling_norm: f32,
    candidates: &[(f32, &SeenMotif)],
) -> f32 {
    if candidates.is_empty() {
        return 0.0;
    }

    let state_scale = rolling_norm * mean_absolute_value(signature);
    let depth_scale = depth as f32 / (depth + 1) as f32;
    let local_similarity = candidates
        .iter()
        .map(|(distance, _)| *distance)
        .sum::<f32>()
        / candidates.len() as f32;
    let nearest_distance = candidates
        .iter()
        .map(|(distance, _)| *distance)
        .min_by(|left, right| left.total_cmp(right))
        .unwrap_or(local_similarity);
    let local_selectivity = local_similarity / (local_similarity + nearest_distance);

    harmonic_mean(state_scale, local_similarity) * depth_scale * local_selectivity
}

fn signature_distance(left: &[f32], right: &[f32]) -> f32 {
    let width = left.len().min(right.len());
    if width == 0 {
        return 0.0;
    }

    left.iter()
        .zip(right.iter())
        .take(width)
        .map(|(left, right)| (left - right).abs())
        .sum::<f32>()
        / width as f32
}

fn normalized_l2(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let sum_sq = values.iter().map(|value| value * value).sum::<f32>();
    (sum_sq / values.len() as f32).sqrt()
}

fn mean_absolute_value(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
}

fn harmonic_mean(left: f32, right: f32) -> f32 {
    let sum = left + right;
    if sum == 0.0 {
        0.0
    } else {
        (left * right / sum) + (left * right / sum)
    }
}

fn quantize_state_signal(rolling_norm: f32, values: &[f32]) -> u16 {
    let score = rolling_norm * (1.0 + mean_absolute_value(values));
    let scaled = (score * 4096.0).round();
    scaled.clamp(0.0, u16::MAX as f32) as u16
}

fn quantize_unsigned_bucket(value: f32, scale: f32) -> u8 {
    let scaled = (value * scale).round();
    scaled.clamp(0.0, u8::MAX as f32) as u8
}

fn quantize_signed_bucket(value: f32, scale: f32, clamp: i8) -> i8 {
    let scaled = (value * scale).round();
    scaled.clamp(-(clamp as f32), clamp as f32) as i8
}

impl StateSignature {
    fn from_values(values: &[f32]) -> Self {
        let rolling_norm = normalized_l2(values);
        let mean_abs = mean_absolute_value(values);
        let mut prefix_bins = [0; STATE_SIGNATURE_PREFIX_WIDTH];
        for (index, slot) in prefix_bins.iter_mut().enumerate() {
            let value = values.get(index).copied().unwrap_or_default();
            *slot = quantize_signed_bucket(value, 4.0, 7);
        }
        let mut suffix_bins = [0; STATE_SIGNATURE_SUFFIX_WIDTH];
        let tail_start = values.len().saturating_sub(STATE_SIGNATURE_SUFFIX_WIDTH);
        for (index, slot) in suffix_bins.iter_mut().enumerate() {
            let value = values.get(tail_start + index).copied().unwrap_or_default();
            *slot = quantize_signed_bucket(value, 4.0, 7);
        }

        Self {
            state_bin: quantize_state_signal(rolling_norm, values),
            norm_bin: quantize_unsigned_bucket(rolling_norm, 32.0),
            mean_abs_bin: quantize_unsigned_bucket(mean_abs, 32.0),
            prefix_bins,
            suffix_bins,
        }
    }
}

#[cfg(test)]
mod split_tests {
    use super::{atom_split_point, split_point, SplitPolicy};
    use crate::faceoff::scan_lexemes;

    #[test]
    fn boundary_aware_split_prefers_newline_boundaries() {
        let input = "AAAAAAAAAAAA\n\nBBBBBBBBBBBB";
        let split = split_point(input.as_bytes(), SplitPolicy::BoundaryAware).unwrap();
        let prev = input[..split].chars().next_back().unwrap_or('\0');
        let next = input[split..].chars().next().unwrap_or('\0');

        assert!(prev == '\n' || next == '\n');
    }

    #[test]
    fn boundary_aware_split_prefers_code_structure_over_midpoint() {
        let input =
            "fn_render_home_identifier_longtail(){AUTH_PROVIDER_2026=1;\nprintln!(\"ok\");\n}\n";
        let boundary = split_point(input.as_bytes(), SplitPolicy::BoundaryAware).unwrap();
        let prev = input[..boundary].chars().next_back().unwrap_or('\0');
        let next = input[boundary..].chars().next().unwrap_or('\0');

        assert!(
            matches!(prev, '\n' | '{' | ';' | ')' | ':')
                || matches!(next, '\n' | '{' | ';' | '(' | ':')
        );
    }

    #[test]
    fn boundary_aware_split_falls_back_when_no_boundary_exists() {
        let input = "abcdefghij";
        let balanced = split_point(input.as_bytes(), SplitPolicy::Balanced).unwrap();
        let boundary = split_point(input.as_bytes(), SplitPolicy::BoundaryAware).unwrap();

        assert_eq!(balanced, boundary);
    }

    #[test]
    fn split_point_preserves_utf8_boundaries() {
        let input = "🙂alpha\nbeta語";
        let split = split_point(input.as_bytes(), SplitPolicy::BoundaryAware).unwrap();

        assert!(std::str::from_utf8(&input.as_bytes()[..split]).is_ok());
        assert!(std::str::from_utf8(&input.as_bytes()[split..]).is_ok());
    }

    #[test]
    fn syntax_aware_atom_split_prefers_line_start_declaration_boundaries() {
        let input = "let auth_provider = 1;\nfn render_home_screen() {\n    println!(\"ok\");\n}\n";
        let atoms = scan_lexemes(input, 0);
        let split = atom_split_point(&atoms, SplitPolicy::SyntaxAware).unwrap();
        let split_start = atoms[split].start;
        let right = &input[split_start..];

        assert!(
            right.starts_with("fn ") || right.starts_with("\nfn "),
            "syntax-aware split chose right segment {:?}",
            right
        );
    }

    #[test]
    fn syntax_aware_atom_split_prefers_markdown_heading_boundaries() {
        let input = "paragraph text paragraph text.\n## Next Section\n- item one\n- item two\n";
        let atoms = scan_lexemes(input, 0);
        let split = atom_split_point(&atoms, SplitPolicy::SyntaxAware).unwrap();
        let split_start = atoms[split].start;
        let right = &input[split_start..];

        assert!(
            right.starts_with("## ")
                || right.starts_with("\n## ")
                || right.starts_with("# ")
                || right.starts_with("\n# ")
                || right.starts_with("- ")
                || right.starts_with("\n- "),
            "syntax-aware split chose right segment {:?}",
            right
        );
    }

    #[test]
    fn syntax_aware_atom_split_preserves_utf8_boundaries() {
        let input = "## 概要\n- 項目🙂\nfn render_home() {\n    return;\n}\n";
        let atoms = scan_lexemes(input, 0);
        let split = atom_split_point(&atoms, SplitPolicy::SyntaxAware).unwrap();
        let split_start = atoms[split].start;

        assert!(std::str::from_utf8(&input.as_bytes()[..split_start]).is_ok());
        assert!(std::str::from_utf8(&input.as_bytes()[split_start..]).is_ok());
    }
}
