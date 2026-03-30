use burn::tensor::{backend::Backend, ElementConversion, Int, Tensor, TensorData};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::error::FractalError;

pub const PAD_TOKEN: usize = 0;
pub const BOS_TOKEN: i64 = 1;
pub const EOS_TOKEN: i64 = 2;
pub const ROOT_TOKEN: i64 = 3;
pub const CLAUSE_OPEN_TOKEN: i64 = 4;
pub const CLAUSE_CLOSE_TOKEN: i64 = 5;
pub const REL_TOKEN: i64 = 6;
pub const NOUN_BASE: i64 = 7;
pub const VERB_BASE: i64 = 10;
pub const DEPTH_BASE: i64 = 13;
pub const GRID_TOKEN: i64 = 21;
pub const ROW_TOKEN: i64 = 22;
pub const CELL_BASE: i64 = 23;
pub const SENTENCE_FAMILY_TOKEN: i64 = 34;
pub const ARC_FAMILY_TOKEN: i64 = 35;
pub const MIN_VOCAB_SIZE: usize = ARC_FAMILY_TOKEN as usize + 1;
pub const MIN_SEQUENCE_LEN: usize = 11;

const SENTENCE_TRAIN_MAX_DEPTH: usize = 4;
const SENTENCE_EVAL_MIN_DEPTH: usize = 5;
const SENTENCE_EVAL_MAX_DEPTH: usize = 8;
const GRID_TRAIN_MAX_DEPTH: usize = 3;
const GRID_EVAL_MIN_DEPTH: usize = 4;
const GRID_EVAL_MAX_DEPTH: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TaskFamily {
    RecursiveSentence,
    ArcGrid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DatasetSplit {
    Train,
    Eval,
}

#[derive(Clone, Debug)]
pub struct GeneratorConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub train_examples_per_family: usize,
    pub eval_examples_per_family: usize,
    pub seed: u64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            vocab_size: 64,
            max_seq_len: 128,
            train_examples_per_family: 96,
            eval_examples_per_family: 32,
            seed: 42,
        }
    }
}

impl GeneratorConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        if self.vocab_size < MIN_VOCAB_SIZE {
            return Err(FractalError::InvalidConfig(format!(
                "vocab_size must be at least {MIN_VOCAB_SIZE} to encode reserved tokens"
            )));
        }
        if self.max_seq_len < MIN_SEQUENCE_LEN {
            return Err(FractalError::InvalidConfig(
                format!(
                    "max_seq_len must be at least {MIN_SEQUENCE_LEN} to encode the smallest recursive task"
                ),
            ));
        }
        if self.train_examples_per_family == 0 {
            return Err(FractalError::InvalidConfig(
                "train_examples_per_family must be greater than zero".into(),
            ));
        }
        if self.eval_examples_per_family == 0 {
            return Err(FractalError::InvalidConfig(
                "eval_examples_per_family must be greater than zero".into(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct RawExample {
    pub tokens: Vec<i64>,
}

#[derive(Debug)]
pub struct TokenBatch<B: Backend> {
    pub input_ids: Tensor<B, 2, Int>,
    pub target_ids: Tensor<B, 2, Int>,
    pub family: TaskFamily,
}

#[derive(Clone, Debug)]
pub struct SimpleHierarchicalGenerator {
    pub config: GeneratorConfig,
    train_sentences: Vec<RawExample>,
    train_grids: Vec<RawExample>,
    eval_sentences: Vec<RawExample>,
    eval_grids: Vec<RawExample>,
}

impl SimpleHierarchicalGenerator {
    pub fn new(config: GeneratorConfig) -> Result<Self, FractalError> {
        config.validate()?;
        let mut train_rng = StdRng::seed_from_u64(config.seed);
        let mut eval_rng = StdRng::seed_from_u64(config.seed.wrapping_add(1));

        let sentence_train_schedule = train_depth_schedule(
            config.max_seq_len,
            SENTENCE_TRAIN_MAX_DEPTH,
            sentence_token_budget,
        )?;
        let grid_train_schedule =
            train_depth_schedule(config.max_seq_len, GRID_TRAIN_MAX_DEPTH, grid_token_budget)?;
        let sentence_eval_schedule = eval_depth_schedule(
            config.max_seq_len,
            SENTENCE_EVAL_MIN_DEPTH,
            SENTENCE_EVAL_MAX_DEPTH,
            sentence_token_budget,
        )?;
        let grid_eval_schedule = eval_depth_schedule(
            config.max_seq_len,
            GRID_EVAL_MIN_DEPTH,
            GRID_EVAL_MAX_DEPTH,
            grid_token_budget,
        )?;

        let train_sentences = build_examples(
            &mut train_rng,
            config.train_examples_per_family,
            sentence_train_schedule,
            build_sentence_tokens,
        );
        let train_grids = build_examples(
            &mut train_rng,
            config.train_examples_per_family,
            grid_train_schedule,
            build_grid_tokens,
        );
        let eval_sentences = build_examples(
            &mut eval_rng,
            config.eval_examples_per_family,
            sentence_eval_schedule,
            build_sentence_tokens,
        );
        let eval_grids = build_examples(
            &mut eval_rng,
            config.eval_examples_per_family,
            grid_eval_schedule,
            build_grid_tokens,
        );

        ensure_examples_fit(&train_sentences, config.max_seq_len, "train sentence")?;
        ensure_examples_fit(&train_grids, config.max_seq_len, "train grid")?;
        ensure_examples_fit(&eval_sentences, config.max_seq_len, "eval sentence")?;
        ensure_examples_fit(&eval_grids, config.max_seq_len, "eval grid")?;

        Ok(Self {
            config,
            train_sentences,
            train_grids,
            eval_sentences,
            eval_grids,
        })
    }

    pub fn batch_for<B: Backend>(
        &self,
        family: TaskFamily,
        split: DatasetSplit,
        batch_index: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TokenBatch<B>, FractalError> {
        let dataset = self.dataset(family, split);
        let start = (batch_index * batch_size) % dataset.len();
        self.dataset_to_batch(dataset, family, start, batch_size, device)
    }

    pub fn eval_batches_for<B: Backend>(
        &self,
        family: TaskFamily,
        batch_size: usize,
        limit: usize,
        device: &B::Device,
    ) -> Result<Vec<TokenBatch<B>>, FractalError> {
        (0..limit)
            .map(|index| self.batch_for(family, DatasetSplit::Eval, index, batch_size, device))
            .collect()
    }

    pub fn max_tokens_for(&self, family: TaskFamily, split: DatasetSplit) -> usize {
        self.dataset(family, split)
            .iter()
            .map(|example| example.tokens.len())
            .max()
            .unwrap_or_default()
    }

    fn dataset(&self, family: TaskFamily, split: DatasetSplit) -> &[RawExample] {
        match (family, split) {
            (TaskFamily::RecursiveSentence, DatasetSplit::Train) => &self.train_sentences,
            (TaskFamily::RecursiveSentence, DatasetSplit::Eval) => &self.eval_sentences,
            (TaskFamily::ArcGrid, DatasetSplit::Train) => &self.train_grids,
            (TaskFamily::ArcGrid, DatasetSplit::Eval) => &self.eval_grids,
        }
    }

    fn dataset_to_batch<B: Backend>(
        &self,
        dataset: &[RawExample],
        family: TaskFamily,
        start: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Result<TokenBatch<B>, FractalError> {
        let seq_len = self.config.max_seq_len;
        let mut input_flat = Vec::with_capacity(batch_size * seq_len);
        let mut target_flat = Vec::with_capacity(batch_size * seq_len);

        for offset in 0..batch_size {
            let example = &dataset[(start + offset) % dataset.len()];
            if example.tokens.len() > seq_len {
                return Err(FractalError::InvalidState(format!(
                    "example length {} exceeds configured max_seq_len {}",
                    example.tokens.len(),
                    seq_len
                )));
            }
            let mut padded = vec![PAD_TOKEN as i64; seq_len];
            for (index, token) in example.tokens.iter().enumerate() {
                padded[index] = *token;
            }

            let mut shifted = padded.clone();
            shifted.rotate_left(1);
            if let Some(last) = shifted.last_mut() {
                *last = PAD_TOKEN as i64;
            }

            input_flat.extend(
                padded
                    .iter()
                    .copied()
                    .map(|token| token.elem::<B::IntElem>()),
            );
            target_flat.extend(
                shifted
                    .iter()
                    .copied()
                    .map(|token| token.elem::<B::IntElem>()),
            );
        }

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_flat, [batch_size, seq_len]),
            device,
        );
        let target_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_flat, [batch_size, seq_len]),
            device,
        );

        Ok(TokenBatch {
            input_ids,
            target_ids,
            family,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DepthSchedule {
    min_depth: usize,
    max_depth: usize,
}

impl DepthSchedule {
    fn new(min_depth: usize, max_depth: usize) -> Result<Self, FractalError> {
        if min_depth == 0 || max_depth == 0 || min_depth > max_depth {
            return Err(FractalError::InvalidConfig(format!(
                "invalid depth schedule {min_depth}..={max_depth}"
            )));
        }

        Ok(Self {
            min_depth,
            max_depth,
        })
    }

    fn depth_for(self, index: usize) -> usize {
        self.min_depth + (index % self.len())
    }

    fn len(self) -> usize {
        self.max_depth - self.min_depth + 1
    }
}

fn build_examples(
    rng: &mut StdRng,
    count: usize,
    schedule: DepthSchedule,
    builder: fn(&mut StdRng, usize) -> Vec<i64>,
) -> Vec<RawExample> {
    (0..count)
        .map(|index| RawExample {
            tokens: builder(rng, schedule.depth_for(index)),
        })
        .collect()
}

fn ensure_examples_fit(
    dataset: &[RawExample],
    max_seq_len: usize,
    label: &'static str,
) -> Result<(), FractalError> {
    if let Some(example) = dataset
        .iter()
        .find(|example| example.tokens.len() > max_seq_len)
    {
        return Err(FractalError::InvalidConfig(format!(
            "{label} example length {} exceeds max_seq_len {max_seq_len}",
            example.tokens.len()
        )));
    }

    Ok(())
}

fn train_depth_schedule(
    max_seq_len: usize,
    preferred_max_depth: usize,
    token_budget: fn(usize) -> usize,
) -> Result<DepthSchedule, FractalError> {
    let max_depth = max_supported_depth(max_seq_len, preferred_max_depth, token_budget)?;
    DepthSchedule::new(1, max_depth)
}

fn eval_depth_schedule(
    max_seq_len: usize,
    preferred_min_depth: usize,
    preferred_max_depth: usize,
    token_budget: fn(usize) -> usize,
) -> Result<DepthSchedule, FractalError> {
    let max_depth = max_supported_depth(max_seq_len, preferred_max_depth, token_budget)?;
    DepthSchedule::new(preferred_min_depth.min(max_depth), max_depth)
}

fn max_supported_depth(
    max_seq_len: usize,
    preferred_max_depth: usize,
    token_budget: fn(usize) -> usize,
) -> Result<usize, FractalError> {
    (1..=preferred_max_depth)
        .rev()
        .find(|depth| token_budget(*depth) <= max_seq_len)
        .ok_or_else(|| {
            FractalError::InvalidConfig(format!(
                "max_seq_len {max_seq_len} cannot encode the requested recursive task"
            ))
        })
}

fn sentence_token_budget(depth: usize) -> usize {
    6 * depth + 4
}

fn grid_token_budget(depth: usize) -> usize {
    let width = 1usize << depth;
    5 + width * width + width
}

fn build_sentence_tokens(rng: &mut StdRng, depth: usize) -> Vec<i64> {
    let mut tokens = vec![
        BOS_TOKEN,
        SENTENCE_FAMILY_TOKEN,
        DEPTH_BASE + (depth as i64 - 1),
        ROOT_TOKEN,
    ];
    build_clause(rng, depth, &mut tokens);
    tokens.push(EOS_TOKEN);
    tokens
}

fn build_clause(rng: &mut StdRng, depth: usize, tokens: &mut Vec<i64>) {
    tokens.push(CLAUSE_OPEN_TOKEN);
    tokens.push(NOUN_BASE + rng.gen_range(0..3) as i64);
    if depth > 1 {
        tokens.push(REL_TOKEN);
        build_clause(rng, depth - 1, tokens);
    }
    if rng.gen_bool(0.35) {
        tokens.push(NOUN_BASE + rng.gen_range(0..3) as i64);
    }
    tokens.push(VERB_BASE + rng.gen_range(0..3) as i64);
    tokens.push(CLAUSE_CLOSE_TOKEN);
}

fn build_grid_tokens(rng: &mut StdRng, depth: usize) -> Vec<i64> {
    let start_value = rng.gen_range(0..4) as u8;
    let grid = expand_grid(vec![vec![start_value]], depth);
    let mut tokens = vec![
        BOS_TOKEN,
        ARC_FAMILY_TOKEN,
        DEPTH_BASE + (depth as i64 - 1),
        GRID_TOKEN,
    ];

    for row in grid {
        for cell in row {
            tokens.push(CELL_BASE + cell as i64);
        }
        tokens.push(ROW_TOKEN);
    }
    tokens.push(EOS_TOKEN);
    tokens
}

fn expand_grid(grid: Vec<Vec<u8>>, depth: usize) -> Vec<Vec<u8>> {
    if depth == 0 {
        return grid;
    }

    let rows = grid.len();
    let cols = grid[0].len();
    let mut expanded = vec![vec![0u8; cols * 2]; rows * 2];

    for (r, row) in grid.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            let base = *value;
            expanded[r * 2][c * 2] = base;
            expanded[r * 2][c * 2 + 1] = (base + 1) % 4;
            expanded[r * 2 + 1][c * 2] = (base + 2) % 4;
            expanded[r * 2 + 1][c * 2 + 1] = (base + 3) % 4;
        }
    }

    expand_grid(expanded, depth - 1)
}
