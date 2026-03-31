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
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

use crate::{B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot};

const DEFAULT_LEVELS: usize = 3;
const TRACKER_REMINDER: &str =
    "Reminder: update docs/tokenizer-tracker.md with the latest tokenizer results.";

#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub dim: usize,
    pub levels: usize,
    pub max_depth: usize,
    pub seed: u64,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            levels: DEFAULT_LEVELS,
            max_depth: 6,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenRecord {
    pub depth: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub token: String,
}

#[derive(Clone, Debug)]
pub struct PrimitiveRunSummary {
    pub primitive: &'static str,
    pub produced: usize,
    pub tokens: Vec<TokenRecord>,
}

#[derive(Clone, Copy)]
pub struct PrimitiveFactory<B: Backend> {
    pub name: &'static str,
    pub build: fn(TokenizerConfig, &B::Device) -> Box<dyn FractalRule<B>>,
}

impl<B: Backend> PrimitiveFactory<B> {
    pub fn new(
        name: &'static str,
        build: fn(TokenizerConfig, &B::Device) -> Box<dyn FractalRule<B>>,
    ) -> Self {
        if let Err(error) = validate_tokenizer_primitive_name(name) {
            panic!("invalid tokenizer primitive name `{name}`: {error}");
        }

        Self { name, build }
    }
}

pub struct RecursiveTokenizer {
    config: TokenizerConfig,
}

#[derive(Clone, Copy)]
struct Segment<'a> {
    bytes: &'a [u8],
    start: usize,
    depth: usize,
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
        let state = FractalState::zeros(rule.state_layout(), 1, rule.hidden_dim(), device)?;
        let mut tokens = Vec::new();
        let root = Segment {
            bytes: text.as_bytes(),
            start: 0,
            depth: 0,
        };
        self.tokenize_segment(rule, state, root, &mut tokens, device)?;
        Ok(tokens)
    }

    pub fn run_factory<B: Backend>(
        &self,
        text: &str,
        device: &B::Device,
        factory: PrimitiveFactory<B>,
    ) -> Result<PrimitiveRunSummary, FractalError> {
        validate_tokenizer_primitive_name(factory.name)?;
        let rule = (factory.build)(self.config, device);
        let tokens = self.tokenize(rule.as_ref(), text, device)?;
        Ok(PrimitiveRunSummary {
            primitive: factory.name,
            produced: tokens.len(),
            tokens,
        })
    }

    fn tokenize_segment<B: Backend>(
        &self,
        rule: &dyn FractalRule<B>,
        state: FractalState<B>,
        segment: Segment<'_>,
        tokens: &mut Vec<TokenRecord>,
        device: &B::Device,
    ) -> Result<(), FractalError> {
        if segment.bytes.is_empty() {
            return Ok(());
        }

        let features = segment_features::<B>(segment.bytes, self.config.dim, device);
        let next_state = rule.apply(&state, &features)?;
        let summary = summarize_readout(&next_state.readout(), segment.bytes, segment.depth)?;
        let end = segment.start + segment.bytes.len();
        tokens.push(TokenRecord {
            depth: segment.depth,
            start: segment.start,
            end,
            text: String::from_utf8_lossy(segment.bytes).into_owned(),
            token: summary,
        });

        if segment.depth + 1 >= self.config.max_depth || segment.bytes.len() <= 1 {
            return Ok(());
        }

        if let Some(split) = split_point(segment.bytes) {
            let (left, right) = segment.bytes.split_at(split);
            self.tokenize_segment(
                rule,
                next_state.clone(),
                Segment {
                    bytes: left,
                    start: segment.start,
                    depth: segment.depth + 1,
                },
                tokens,
                device,
            )?;
            self.tokenize_segment(
                rule,
                next_state,
                Segment {
                    bytes: right,
                    start: segment.start + split,
                    depth: segment.depth + 1,
                },
                tokens,
                device,
            )?;
        }

        Ok(())
    }
}

pub fn revived_primitive_factories<B: Backend>() -> [PrimitiveFactory<B>; 5] {
    [
        PrimitiveFactory::new("b1_fractal_gated_v1", seeded_b1_fractal_gated::<B>),
        PrimitiveFactory::new("p1_fractal_hybrid_v1", seeded_p1_fractal_hybrid::<B>),
        PrimitiveFactory::new("p2_mandelbrot_v1", seeded_p2_mandelbrot::<B>),
        PrimitiveFactory::new(
            "b3_fractal_hierarchical_v1",
            seeded_b3_fractal_hierarchical::<B>,
        ),
        PrimitiveFactory::new("b4_universal_v1", seeded_b4_universal::<B>),
    ]
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
    let mut rule = P1FractalHybrid::new(config.dim, device);
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

fn split_point(bytes: &[u8]) -> Option<usize> {
    if bytes.len() <= 1 {
        return None;
    }

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

fn summarize_readout<B: Backend>(
    readout: &Tensor<B, 2>,
    bytes: &[u8],
    depth: usize,
) -> Result<String, FractalError> {
    let values = tensor_data_to_vec::<f32>(readout.clone().into_data(), "token readout")?;
    let prefix = values.iter().take(8).copied().collect::<Vec<_>>();
    let mut digest = 1469598103934665603u64;
    for value in prefix {
        let quantized = (value * 1000.0).round() as i64;
        digest ^= quantized as u64;
        digest = digest.wrapping_mul(1099511628211);
    }
    Ok(format!("d{depth}-n{}-{digest:016x}", bytes.len()))
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
