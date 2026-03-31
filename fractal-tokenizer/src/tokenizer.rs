use burn::tensor::{backend::Backend, Element, Tensor, TensorData};
use fractal_core::{
    error::FractalError,
    rule_trait::FractalRule,
    state::{FractalState, StateLayout},
};

use crate::{B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot};

const DEFAULT_LEVELS: usize = 3;

#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub dim: usize,
    pub levels: usize,
    pub max_depth: usize,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            dim: 64,
            levels: DEFAULT_LEVELS,
            max_depth: 6,
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
        PrimitiveFactory {
            name: "b1_fractal_gated",
            build: |config, device| Box::new(B1FractalGated::new(config.dim, device)),
        },
        PrimitiveFactory {
            name: "p1_fractal_hybrid",
            build: |config, device| Box::new(P1FractalHybrid::new(config.dim, device)),
        },
        PrimitiveFactory {
            name: "p2_mandelbrot",
            build: |config, device| Box::new(P2Mandelbrot::new(config.dim, device)),
        },
        PrimitiveFactory {
            name: "b3_fractal_hierarchical",
            build: |config, device| {
                Box::new(B3FractalHierarchical::new(
                    config.dim,
                    config.levels,
                    device,
                ))
            },
        },
        PrimitiveFactory {
            name: "b4_universal",
            build: |config, device| Box::new(B4Universal::new(config.dim, config.levels, device)),
        },
    ]
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
