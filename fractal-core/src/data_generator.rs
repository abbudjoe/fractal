use burn::tensor::{backend::Backend, Int, Tensor, TensorData};
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
        if self.max_seq_len == 0 {
            return Err(FractalError::InvalidConfig(
                "max_seq_len must be greater than zero".into(),
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

        let train_sentences = (0..config.train_examples_per_family)
            .map(|index| RawExample {
                tokens: build_sentence_tokens(&mut train_rng, 1 + (index % 4)),
            })
            .collect();

        let train_grids = (0..config.train_examples_per_family)
            .map(|index| RawExample {
                tokens: build_grid_tokens(&mut train_rng, 1 + (index % 3)),
            })
            .collect();

        let eval_sentences = (0..config.eval_examples_per_family)
            .map(|index| RawExample {
                tokens: build_sentence_tokens(&mut eval_rng, 5 + (index % 4)),
            })
            .collect();

        let eval_grids = (0..config.eval_examples_per_family)
            .map(|index| RawExample {
                tokens: build_grid_tokens(&mut eval_rng, 4 + (index % 2)),
            })
            .collect();

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
    ) -> TokenBatch<B> {
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
    ) -> Vec<TokenBatch<B>> {
        (0..limit)
            .map(|index| self.batch_for(family, DatasetSplit::Eval, index, batch_size, device))
            .collect()
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
    ) -> TokenBatch<B> {
        let seq_len = self.config.max_seq_len;
        let mut input_flat = Vec::with_capacity(batch_size * seq_len);
        let mut target_flat = Vec::with_capacity(batch_size * seq_len);

        for offset in 0..batch_size {
            let example = &dataset[(start + offset) % dataset.len()];
            let mut padded = vec![PAD_TOKEN as i64; seq_len];
            for (index, token) in example.tokens.iter().take(seq_len).enumerate() {
                padded[index] = *token;
            }

            let mut shifted = padded.clone();
            shifted.rotate_left(1);
            if let Some(last) = shifted.last_mut() {
                *last = PAD_TOKEN as i64;
            }

            input_flat.extend_from_slice(&padded);
            target_flat.extend_from_slice(&shifted);
        }

        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_flat, [batch_size, seq_len]),
            device,
        );
        let target_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_flat, [batch_size, seq_len]),
            device,
        );

        TokenBatch {
            input_ids,
            target_ids,
            family,
        }
    }
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
