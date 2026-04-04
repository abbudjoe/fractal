use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    prelude::ElementConversion,
    record::{BinFileRecorder, FullPrecisionSettings, Recorder},
    tensor::{backend::AutodiffBackend, backend::Backend, Int, Tensor, TensorData},
};
use serde::{Deserialize, Serialize};

use fractal_core::{error::FractalError, TaskFamily, TokenBatch, PAD_TOKEN};

use crate::{
    build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig,
};

pub const BYTE_LEVEL_PAD_TOKEN: usize = PAD_TOKEN;
pub const BYTE_LEVEL_VOCAB_SIZE: usize = 257;
pub const DEFAULT_V2_SMOKE_SEQ_LEN: usize = 32;
pub const DEFAULT_V2_SMOKE_WINDOW_STRIDE: usize = DEFAULT_V2_SMOKE_SEQ_LEN;
pub const DEFAULT_V2_SMOKE_BATCH_SIZE: usize = 1;
pub const DEFAULT_V2_SMOKE_TRAIN_STEPS: usize = 2;
pub const DEFAULT_V2_SMOKE_EVAL_BATCHES: usize = 1;
pub const DEFAULT_V2_SMOKE_EVAL_HOLDOUT_EVERY: usize = 10;
pub const DEFAULT_V2_SMOKE_LEARNING_RATE: f64 = 1e-3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ByteLevelVocabularyContract {
    pub pad_token: usize,
    pub vocab_size: usize,
}

impl Default for ByteLevelVocabularyContract {
    fn default() -> Self {
        Self {
            pad_token: BYTE_LEVEL_PAD_TOKEN,
            vocab_size: BYTE_LEVEL_VOCAB_SIZE,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SmokeTrainConfig {
    pub corpus_paths: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub model: BaselineV2SyntheticModelConfig,
    pub seq_len: usize,
    pub window_stride: usize,
    pub batch_size: usize,
    pub train_steps: usize,
    pub eval_batches: usize,
    pub eval_holdout_every: usize,
    pub learning_rate: f64,
    pub vocabulary: ByteLevelVocabularyContract,
}

impl V2SmokeTrainConfig {
    pub fn new(corpus_paths: Vec<PathBuf>, output_dir: PathBuf) -> Self {
        Self {
            corpus_paths,
            output_dir,
            model: baseline_v2_byte_level_smoke_model_config(),
            seq_len: DEFAULT_V2_SMOKE_SEQ_LEN,
            window_stride: DEFAULT_V2_SMOKE_WINDOW_STRIDE,
            batch_size: DEFAULT_V2_SMOKE_BATCH_SIZE,
            train_steps: DEFAULT_V2_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V2_SMOKE_EVAL_BATCHES,
            eval_holdout_every: DEFAULT_V2_SMOKE_EVAL_HOLDOUT_EVERY,
            learning_rate: DEFAULT_V2_SMOKE_LEARNING_RATE,
            vocabulary: ByteLevelVocabularyContract::default(),
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.corpus_paths.is_empty() {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.corpus_paths must include at least one file".to_string(),
            ));
        }
        if self.seq_len == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.seq_len must be greater than zero".to_string(),
            ));
        }
        if self.window_stride == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.window_stride must be greater than zero".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.batch_size must be greater than zero".to_string(),
            ));
        }
        if self.train_steps == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.train_steps must be greater than zero".to_string(),
            ));
        }
        if self.eval_batches == 0 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.eval_batches must be greater than zero".to_string(),
            ));
        }
        if self.eval_holdout_every < 2 {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.eval_holdout_every must be at least 2".to_string(),
            ));
        }
        if !(self.learning_rate.is_finite() && self.learning_rate > 0.0) {
            return Err(FractalError::InvalidConfig(
                "v2_smoke_train.learning_rate must be finite and greater than zero".to_string(),
            ));
        }
        if self.vocabulary.pad_token != BYTE_LEVEL_PAD_TOKEN {
            return Err(FractalError::InvalidConfig(format!(
                "v2_smoke_train.vocabulary.pad_token must remain {BYTE_LEVEL_PAD_TOKEN} for byte-level smoke training, got {}",
                self.vocabulary.pad_token
            )));
        }
        if self.vocabulary.vocab_size != BYTE_LEVEL_VOCAB_SIZE {
            return Err(FractalError::InvalidConfig(format!(
                "v2_smoke_train.vocabulary.vocab_size must remain {BYTE_LEVEL_VOCAB_SIZE} for byte-level smoke training, got {}",
                self.vocabulary.vocab_size
            )));
        }
        if self.model.vocab_size != self.vocabulary.vocab_size {
            return Err(FractalError::InvalidConfig(format!(
                "v2_smoke_train.model.vocab_size ({}) must match byte-level vocab size ({})",
                self.model.vocab_size, self.vocabulary.vocab_size
            )));
        }
        self.model.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2SmokeCorpusStats {
    pub files: Vec<PathBuf>,
    pub total_bytes: usize,
    pub total_sequences: usize,
    pub train_sequences: usize,
    pub eval_sequences: usize,
    pub seq_len: usize,
    pub window_stride: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct V2SmokeTrainStepReport {
    pub step: usize,
    pub learning_rate: f64,
    pub train_loss: f64,
    pub train_perplexity: f64,
    pub seen_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SmokeEvalMetrics {
    pub batch_count: usize,
    pub mean_loss: f64,
    pub perplexity: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct V2SmokeCheckpointArtifacts {
    pub directory: PathBuf,
    pub final_model_path: PathBuf,
    pub best_model_path: PathBuf,
    pub final_optimizer_path: PathBuf,
    pub best_optimizer_path: PathBuf,
    pub report_path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum V2SmokeCheckpointKind {
    InitialEval,
    FinalEval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2SmokeTrainReport {
    pub config: V2SmokeTrainConfig,
    pub corpus: V2SmokeCorpusStats,
    pub initial_eval: V2SmokeEvalMetrics,
    pub final_eval: V2SmokeEvalMetrics,
    pub best_eval: V2SmokeEvalMetrics,
    pub best_checkpoint_kind: V2SmokeCheckpointKind,
    pub train_steps: Vec<V2SmokeTrainStepReport>,
    pub checkpoint: V2SmokeCheckpointArtifacts,
}

#[derive(Debug)]
pub struct V2SmokeTrainResult<M> {
    pub model: M,
    pub report: V2SmokeTrainReport,
}

pub trait V2SmokeTrainModel<B: Backend>: Module<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError>;
    fn vocab_size(&self) -> usize;
}

impl<B: Backend> V2SmokeTrainModel<B> for BaselineV2SyntheticModel<B> {
    fn forward_logits(&self, input_ids: Tensor<B, 2, Int>) -> Result<Tensor<B, 3>, FractalError> {
        Ok(self.forward(input_ids)?.logits())
    }

    fn vocab_size(&self) -> usize {
        self.shape().vocab_size
    }
}

pub fn baseline_v2_byte_level_smoke_model_config() -> BaselineV2SyntheticModelConfig {
    BaselineV2SyntheticModelConfig::new(BYTE_LEVEL_VOCAB_SIZE, 8)
}

pub fn default_v2_smoke_corpus_paths(
    repo_root: impl AsRef<Path>,
) -> Result<Vec<PathBuf>, FractalError> {
    let repo_root = repo_root.as_ref();
    let mut paths = Vec::new();

    for relative in ["AGENTS.md", "ENGINEERING.md", "README.md"] {
        let path = repo_root.join(relative);
        if path.is_file() {
            paths.push(path);
        }
    }

    collect_markdown_files_recursive(&repo_root.join("docs/specs"), &mut paths)?;
    paths.sort();
    paths.dedup();

    if paths.is_empty() {
        return Err(FractalError::InvalidState(format!(
            "default v2 smoke corpus under {} did not contain any readable markdown files",
            repo_root.display()
        )));
    }

    Ok(paths)
}

pub fn run_baseline_v2_smoke_train<B>(
    config: V2SmokeTrainConfig,
    device: &B::Device,
) -> Result<V2SmokeTrainResult<BaselineV2SyntheticModel<B>>, FractalError>
where
    B: AutodiffBackend,
{
    let model = build_baseline_v2_synthetic_model::<B>(config.model, device)?;
    run_v2_smoke_train_with_model(model, config, device)
}

pub fn run_v2_smoke_train_with_model<B, M>(
    model: M,
    config: V2SmokeTrainConfig,
    device: &B::Device,
) -> Result<V2SmokeTrainResult<M>, FractalError>
where
    B: AutodiffBackend,
    M: V2SmokeTrainModel<B> + AutodiffModule<B> + Module<B> + Clone,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    config.validate()?;
    let corpus = load_byte_level_corpus(&config.corpus_paths)?;
    let sequences =
        byte_sequences_from_corpus(&corpus.files, config.seq_len, config.window_stride)?;
    let (train_sequences, eval_sequences) =
        split_sequences_for_eval(sequences, config.eval_holdout_every)?;
    let train_batches = sequences_into_batches::<B>(train_sequences, config.batch_size, device)?;
    let eval_batches = sequences_into_batches::<B>(eval_sequences, config.batch_size, device)?;
    let criterion = CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![BYTE_LEVEL_PAD_TOKEN]))
        .init(device);
    let initial_optimizer = AdamConfig::new().init::<B, M>();
    let mut optimizer = AdamConfig::new().init::<B, M>();
    let initial_eval = evaluate_model(&model, &criterion, &eval_batches, config.eval_batches)?;
    let initial_model = model.clone();

    let mut model = model;
    let mut seen_tokens = 0usize;
    let mut train_steps = Vec::with_capacity(config.train_steps);

    for step in 0..config.train_steps {
        let batch = &train_batches[step % train_batches.len()];
        let loss = next_token_loss(&model, batch, &criterion)?;
        let train_loss = loss.clone().into_scalar().elem::<f64>();
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        seen_tokens += batch.token_count;
        train_steps.push(V2SmokeTrainStepReport {
            step: step + 1,
            learning_rate: config.learning_rate,
            train_loss,
            train_perplexity: train_loss.exp(),
            seen_tokens,
        });
    }

    let final_eval = evaluate_model(&model, &criterion, &eval_batches, config.eval_batches)?;
    let (best_eval, best_checkpoint_kind, best_model, best_optimizer) =
        if final_eval.mean_loss <= initial_eval.mean_loss {
            (
                final_eval.clone(),
                V2SmokeCheckpointKind::FinalEval,
                None,
                None,
            )
        } else {
            (
                initial_eval.clone(),
                V2SmokeCheckpointKind::InitialEval,
                Some(&initial_model),
                Some(&initial_optimizer),
            )
        };
    let checkpoint =
        persist_smoke_train_artifacts(&model, best_model, &optimizer, best_optimizer, &config)?;
    let corpus_paths = corpus.paths;
    let total_sequences = train_batches
        .iter()
        .map(|batch| batch.input_ids.dims()[0])
        .sum::<usize>()
        + eval_batches
            .iter()
            .map(|batch| batch.input_ids.dims()[0])
            .sum::<usize>();
    let train_sequences = train_batches
        .iter()
        .map(|batch| batch.input_ids.dims()[0])
        .sum::<usize>();
    let eval_sequences = eval_batches
        .iter()
        .map(|batch| batch.input_ids.dims()[0])
        .sum::<usize>();
    let seq_len = train_batches
        .first()
        .map(|batch| batch.input_ids.dims()[1])
        .unwrap_or(config.seq_len);
    let window_stride = config.window_stride;
    let total_bytes = corpus.total_bytes;
    let report = V2SmokeTrainReport {
        config,
        corpus: V2SmokeCorpusStats {
            files: corpus_paths,
            total_bytes,
            total_sequences,
            train_sequences,
            eval_sequences,
            seq_len,
            window_stride,
        },
        initial_eval,
        final_eval,
        best_eval,
        best_checkpoint_kind,
        train_steps,
        checkpoint,
    };
    write_report(&report)?;

    Ok(V2SmokeTrainResult { model, report })
}

#[derive(Debug)]
struct LoadedCorpus {
    paths: Vec<PathBuf>,
    files: Vec<Vec<u8>>,
    total_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ByteSequence {
    input: Vec<i64>,
    target: Vec<i64>,
}

fn load_byte_level_corpus(paths: &[PathBuf]) -> Result<LoadedCorpus, FractalError> {
    let mut loaded_paths = Vec::new();
    let mut files = Vec::new();
    let mut total_bytes = 0usize;

    for path in paths {
        let data = fs::read(path).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read corpus file {}: {error}",
                path.display()
            ))
        })?;
        if data.is_empty() {
            continue;
        }
        total_bytes += data.len();
        loaded_paths.push(path.clone());
        files.push(data);
    }

    if files.is_empty() {
        return Err(FractalError::InvalidConfig(
            "v2 smoke training corpus did not contain any non-empty files".to_string(),
        ));
    }

    Ok(LoadedCorpus {
        paths: loaded_paths,
        files,
        total_bytes,
    })
}

fn byte_sequences_from_corpus(
    files: &[Vec<u8>],
    seq_len: usize,
    window_stride: usize,
) -> Result<Vec<ByteSequence>, FractalError> {
    let required_len = seq_len + 1;
    let mut sequences = Vec::new();

    for bytes in files {
        if bytes.len() < required_len {
            continue;
        }
        for start in (0..=bytes.len() - required_len).step_by(window_stride) {
            let window = &bytes[start..start + required_len];
            let mut input = Vec::with_capacity(seq_len);
            let mut target = Vec::with_capacity(seq_len);
            for index in 0..seq_len {
                input.push((window[index] as i64) + 1);
                target.push((window[index + 1] as i64) + 1);
            }
            sequences.push(ByteSequence { input, target });
        }
    }

    if sequences.len() < 2 {
        return Err(FractalError::InvalidConfig(format!(
            "v2 smoke training corpus must yield at least 2 sequences of length {}, got {}",
            seq_len,
            sequences.len()
        )));
    }

    Ok(sequences)
}

fn split_sequences_for_eval(
    sequences: Vec<ByteSequence>,
    eval_holdout_every: usize,
) -> Result<(Vec<ByteSequence>, Vec<ByteSequence>), FractalError> {
    let mut train = Vec::new();
    let mut eval = Vec::new();

    for (index, sequence) in sequences.into_iter().enumerate() {
        if index % eval_holdout_every == 0 {
            eval.push(sequence);
        } else {
            train.push(sequence);
        }
    }

    if train.is_empty() || eval.is_empty() {
        return Err(FractalError::InvalidConfig(format!(
            "v2 smoke training split must produce both train and eval sequences, got train={} eval={}",
            train.len(),
            eval.len()
        )));
    }

    Ok((train, eval))
}

fn sequences_into_batches<B: Backend>(
    sequences: Vec<ByteSequence>,
    batch_size: usize,
    device: &B::Device,
) -> Result<Vec<TokenBatch<B>>, FractalError> {
    if sequences.is_empty() {
        return Err(FractalError::InvalidConfig(
            "v2 smoke training requires at least one sequence".to_string(),
        ));
    }

    let seq_len = sequences[0].input.len();
    let mut batches = Vec::new();
    for chunk in sequences.chunks(batch_size) {
        let mut input_flat = Vec::with_capacity(chunk.len() * seq_len);
        let mut target_flat = Vec::with_capacity(chunk.len() * seq_len);
        for sequence in chunk {
            input_flat.extend_from_slice(&sequence.input);
            target_flat.extend_from_slice(&sequence.target);
        }
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_flat, [chunk.len(), seq_len]),
            device,
        );
        let target_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_flat, [chunk.len(), seq_len]),
            device,
        );
        batches.push(TokenBatch {
            input_ids,
            target_ids,
            token_count: chunk.len() * seq_len,
            family: TaskFamily::TokenizerBackedText,
        });
    }
    Ok(batches)
}

fn next_token_loss<B, M>(
    model: &M,
    batch: &TokenBatch<B>,
    criterion: &CrossEntropyLoss<B>,
) -> Result<Tensor<B, 1>, FractalError>
where
    B: AutodiffBackend,
    M: V2SmokeTrainModel<B>,
{
    let logits = model.forward_logits(batch.input_ids.clone())?;
    let [batch_size, seq_len, vocab_size] = logits.dims();
    let target_dims = batch.target_ids.dims();
    if target_dims != [batch_size, seq_len] {
        return Err(FractalError::Shape(format!(
            "v2 smoke training target shape mismatch: logits [{batch_size}, {seq_len}, {vocab_size}] vs targets [{}, {}]",
            target_dims[0], target_dims[1]
        )));
    }
    if vocab_size != model.vocab_size() {
        return Err(FractalError::Shape(format!(
            "v2 smoke training vocab mismatch: logits vocab {} vs model vocab {}",
            vocab_size,
            model.vocab_size()
        )));
    }

    Ok(criterion.forward(
        logits.reshape([batch_size * seq_len, vocab_size]),
        batch.target_ids.clone().reshape([batch_size * seq_len]),
    ))
}

fn evaluate_model<B, M>(
    model: &M,
    criterion: &CrossEntropyLoss<B>,
    batches: &[TokenBatch<B>],
    eval_batches: usize,
) -> Result<V2SmokeEvalMetrics, FractalError>
where
    B: AutodiffBackend,
    M: V2SmokeTrainModel<B>,
{
    let take = eval_batches.min(batches.len());
    if take == 0 {
        return Err(FractalError::InvalidConfig(
            "v2 smoke training eval split must include at least one batch".to_string(),
        ));
    }

    let mut total_loss = 0.0f64;
    for batch in &batches[..take] {
        total_loss += next_token_loss(model, batch, criterion)?
            .into_scalar()
            .elem::<f64>();
    }
    let mean_loss = total_loss / take as f64;
    Ok(V2SmokeEvalMetrics {
        batch_count: take,
        mean_loss,
        perplexity: mean_loss.exp(),
    })
}

fn persist_smoke_train_artifacts<B, M>(
    final_model: &M,
    best_model: Option<&M>,
    final_optimizer: &OptimizerAdaptor<Adam, M, B>,
    best_optimizer: Option<&OptimizerAdaptor<Adam, M, B>>,
    config: &V2SmokeTrainConfig,
) -> Result<V2SmokeCheckpointArtifacts, FractalError>
where
    B: AutodiffBackend,
    M: Module<B> + Clone + AutodiffModule<B>,
    <M as AutodiffModule<B>>::InnerModule: Module<B::InnerBackend>,
{
    ensure_empty_output_dir(&config.output_dir)?;

    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let final_model_stem = config.output_dir.join("model");
    let best_model_stem = config.output_dir.join("best-model");
    let final_optimizer_stem = config.output_dir.join("optimizer");
    let best_optimizer_stem = config.output_dir.join("best-optimizer");
    final_model
        .clone()
        .save_file(final_model_stem.clone(), &recorder)
        .map_err(recorder_error)?;
    let best_model_path = if let Some(best_model) = best_model {
        best_model
            .clone()
            .save_file(best_model_stem.clone(), &recorder)
            .map_err(recorder_error)?;
        resolve_written_artifact(&config.output_dir, "best-model")?
    } else {
        resolve_written_artifact(&config.output_dir, "model")?
    };
    recorder
        .record(final_optimizer.to_record(), final_optimizer_stem.clone())
        .map(|_| ())
        .map_err(recorder_error)?;
    let best_optimizer_path = if let Some(best_optimizer) = best_optimizer {
        recorder
            .record(best_optimizer.to_record(), best_optimizer_stem.clone())
            .map(|_| ())
            .map_err(recorder_error)?;
        resolve_written_artifact(&config.output_dir, "best-optimizer")?
    } else {
        resolve_written_artifact(&config.output_dir, "optimizer")?
    };

    Ok(V2SmokeCheckpointArtifacts {
        directory: config.output_dir.clone(),
        final_model_path: resolve_written_artifact(&config.output_dir, "model")?,
        best_model_path,
        final_optimizer_path: resolve_written_artifact(&config.output_dir, "optimizer")?,
        best_optimizer_path,
        report_path: config.output_dir.join("report.json"),
    })
}

fn ensure_empty_output_dir(path: &Path) -> Result<(), FractalError> {
    if path.exists() {
        let mut entries = fs::read_dir(path).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to inspect v2 smoke output directory {}: {error}",
                path.display()
            ))
        })?;
        if entries
            .next()
            .transpose()
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to inspect v2 smoke output directory {}: {error}",
                    path.display()
                ))
            })?
            .is_some()
        {
            return Err(FractalError::InvalidConfig(format!(
                "v2 smoke output directory {} must be empty when provided explicitly",
                path.display()
            )));
        }
    } else {
        fs::create_dir_all(path).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to create v2 smoke output directory {}: {error}",
                path.display()
            ))
        })?;
    }

    Ok(())
}

fn write_report(report: &V2SmokeTrainReport) -> Result<(), FractalError> {
    let payload = serde_json::to_vec_pretty(report).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize v2 smoke training report: {error}"
        ))
    })?;
    fs::write(&report.checkpoint.report_path, payload).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to write v2 smoke training report {}: {error}",
            report.checkpoint.report_path.display()
        ))
    })
}

fn resolve_written_artifact(directory: &Path, prefix: &str) -> Result<PathBuf, FractalError> {
    let mut matches = fs::read_dir(directory)
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to scan artifact directory {}: {error}",
                directory.display()
            ))
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name == prefix || name.starts_with(&format!("{prefix}.")))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches.into_iter().next().ok_or_else(|| {
        FractalError::InvalidState(format!(
            "artifact directory {} did not contain a {} checkpoint artifact",
            directory.display(),
            prefix
        ))
    })
}

fn collect_markdown_files_recursive(
    root: &Path,
    output: &mut Vec<PathBuf>,
) -> Result<(), FractalError> {
    if !root.exists() {
        return Ok(());
    }
    let mut entries = fs::read_dir(root)
        .map_err(|error| {
            FractalError::InvalidState(format!("failed to scan {}: {error}", root.display()))
        })?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_markdown_files_recursive(&path, output)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            output.push(path);
        }
    }
    Ok(())
}

fn recorder_error<E: std::fmt::Display>(error: E) -> FractalError {
    FractalError::InvalidState(format!("checkpoint recorder failed: {error}"))
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::backend::Autodiff;
    use burn::backend::Candle;

    use super::*;

    type TestBackend = Autodiff<Candle<f32, i64>>;

    #[test]
    fn byte_level_sequences_shift_bytes_and_preserve_pad_zero() {
        let sequences = byte_sequences_from_corpus(&[b"abcdefg".to_vec()], 3, 3).unwrap();

        assert_eq!(sequences.len(), 2);
        assert_eq!(BYTE_LEVEL_PAD_TOKEN, 0);
        assert_eq!(sequences[0].input, vec![98, 99, 100]);
        assert_eq!(sequences[0].target, vec![99, 100, 101]);
        assert_eq!(sequences[1].input, vec![101, 102, 103]);
        assert_eq!(sequences[1].target, vec![102, 103, 104]);
    }

    #[test]
    fn smoke_train_persists_checkpoint_and_report() {
        let root = unique_temp_dir("v2-smoke-train-test");
        fs::create_dir_all(&root).unwrap();
        let corpus_a = root.join("a.md");
        let corpus_b = root.join("b.md");
        fs::write(
            &corpus_a,
            "fractal v2 smoke training needs real text windows.\n".repeat(8),
        )
        .unwrap();
        fs::write(
            &corpus_b,
            "sealed leaves and exact reads should learn something.\n".repeat(8),
        )
        .unwrap();
        let output_dir = root.join("out");
        let config = V2SmokeTrainConfig {
            corpus_paths: vec![corpus_a, corpus_b],
            output_dir: output_dir.clone(),
            model: baseline_v2_byte_level_smoke_model_config(),
            seq_len: 16,
            window_stride: 16,
            batch_size: 2,
            train_steps: 2,
            eval_batches: 1,
            eval_holdout_every: 3,
            learning_rate: 1e-3,
            vocabulary: ByteLevelVocabularyContract::default(),
        };

        let device = <TestBackend as Backend>::Device::default();
        let result = run_baseline_v2_smoke_train::<TestBackend>(config, &device).unwrap();

        assert!(result.report.initial_eval.mean_loss.is_finite());
        assert!(result.report.final_eval.mean_loss.is_finite());
        assert!(result.report.checkpoint.final_model_path.exists());
        assert!(result.report.checkpoint.best_model_path.exists());
        assert!(result.report.checkpoint.final_optimizer_path.exists());
        assert!(result.report.checkpoint.best_optimizer_path.exists());
        assert!(result.report.checkpoint.report_path.exists());
        assert_eq!(
            result.report.checkpoint.best_model_path,
            result.report.checkpoint.final_model_path
        );
        assert_eq!(
            result.report.checkpoint.best_optimizer_path,
            result.report.checkpoint.final_optimizer_path
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn smoke_train_rejects_non_empty_output_directory() {
        let root = unique_temp_dir("v2-smoke-train-nonempty");
        fs::create_dir_all(&root).unwrap();
        let corpus = root.join("corpus.md");
        fs::write(
            &corpus,
            "fractal v2 output dirs should be explicit.\n".repeat(12),
        )
        .unwrap();
        let output_dir = root.join("out");
        fs::create_dir_all(&output_dir).unwrap();
        fs::write(output_dir.join("stale.txt"), "stale").unwrap();
        let config = V2SmokeTrainConfig {
            corpus_paths: vec![corpus],
            output_dir: output_dir.clone(),
            model: baseline_v2_byte_level_smoke_model_config(),
            seq_len: 16,
            window_stride: 16,
            batch_size: 2,
            train_steps: 1,
            eval_batches: 1,
            eval_holdout_every: 3,
            learning_rate: 1e-3,
            vocabulary: ByteLevelVocabularyContract::default(),
        };

        let device = <TestBackend as Backend>::Device::default();
        let error = run_baseline_v2_smoke_train::<TestBackend>(config, &device).unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message)
            if message.contains("must be empty")
        ));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn smoke_train_records_initial_best_checkpoint_when_final_eval_regresses() {
        let root = unique_temp_dir("v2-smoke-train-best-checkpoint");
        fs::create_dir_all(&root).unwrap();
        let corpus = root.join("corpus.md");
        fs::write(
            &corpus,
            "ENGINEERING smoke checkpoint regression.\n".repeat(16),
        )
        .unwrap();
        let output_dir = root.join("out");
        let config = V2SmokeTrainConfig {
            corpus_paths: vec![corpus],
            output_dir,
            model: baseline_v2_byte_level_smoke_model_config(),
            seq_len: 16,
            window_stride: 16,
            batch_size: 1,
            train_steps: 1,
            eval_batches: 1,
            eval_holdout_every: 3,
            learning_rate: 100.0,
            vocabulary: ByteLevelVocabularyContract::default(),
        };

        let device = <TestBackend as Backend>::Device::default();
        let result = run_baseline_v2_smoke_train::<TestBackend>(config, &device).unwrap();

        assert_eq!(
            result.report.best_checkpoint_kind,
            V2SmokeCheckpointKind::InitialEval
        );
        assert_ne!(
            result.report.checkpoint.best_model_path,
            result.report.checkpoint.final_model_path
        );
        assert_ne!(
            result.report.checkpoint.best_optimizer_path,
            result.report.checkpoint.final_optimizer_path
        );

        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("{prefix}-{nanos}"))
    }
}
