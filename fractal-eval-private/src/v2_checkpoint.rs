use std::{
    fmt::Display,
    fs,
    path::{Path, PathBuf},
};

use burn::{
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::backend::Backend,
};

use fractal_core::error::FractalError;

use crate::{build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, V2SmokeTrainReport};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2CheckpointSelection {
    Best,
    Final,
}

impl V2CheckpointSelection {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Best => "best",
            Self::Final => "final",
        }
    }

    fn model_path(self, report: &V2SmokeTrainReport) -> &Path {
        match self {
            Self::Best => report.checkpoint.best_model_path.as_path(),
            Self::Final => report.checkpoint.final_model_path.as_path(),
        }
    }
}

#[derive(Debug)]
pub struct LoadedV2CheckpointModel<B: Backend> {
    pub model: BaselineV2SyntheticModel<B>,
    pub report: V2SmokeTrainReport,
    pub report_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub selection: V2CheckpointSelection,
}

pub fn load_v2_smoke_train_report(
    report_path: impl AsRef<Path>,
) -> Result<V2SmokeTrainReport, FractalError> {
    let report_path = report_path.as_ref();
    let payload = fs::read_to_string(report_path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to read v2 smoke train report {}: {error}",
            report_path.display()
        ))
    })?;
    let report = serde_json::from_str::<V2SmokeTrainReport>(&payload).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to parse v2 smoke train report {}: {error}",
            report_path.display()
        ))
    })?;
    report.config.validate()?;
    Ok(report)
}

pub fn load_baseline_v2_checkpoint_model<B: Backend>(
    report_path: impl AsRef<Path>,
    selection: V2CheckpointSelection,
    device: &B::Device,
) -> Result<LoadedV2CheckpointModel<B>, FractalError> {
    let report_path = report_path.as_ref().to_path_buf();
    let report = load_v2_smoke_train_report(&report_path)?;
    let checkpoint_path = resolve_checkpoint_artifact(&report_path, selection.model_path(&report))?;
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = build_baseline_v2_synthetic_model::<B>(report.config.model, device)?
        .load_file(checkpoint_path.clone(), &recorder, device)
        .map_err(recorder_error)?;

    Ok(LoadedV2CheckpointModel {
        model,
        report,
        report_path,
        checkpoint_path,
        selection,
    })
}

fn resolve_checkpoint_artifact(
    report_path: &Path,
    artifact_path: &Path,
) -> Result<PathBuf, FractalError> {
    if artifact_path.is_file() {
        return Ok(artifact_path.to_path_buf());
    }

    let report_dir = report_path.parent().ok_or_else(|| {
        FractalError::InvalidState(format!(
            "v2 smoke train report {} has no parent directory",
            report_path.display()
        ))
    })?;
    let file_name = artifact_path.file_name().ok_or_else(|| {
        FractalError::InvalidState(format!(
            "checkpoint artifact path {} has no file name",
            artifact_path.display()
        ))
    })?;
    let relocated = report_dir.join(file_name);
    if relocated.is_file() {
        return Ok(relocated);
    }

    Err(FractalError::InvalidState(format!(
        "checkpoint artifact {} referenced by {} was not found",
        artifact_path.display(),
        report_path.display()
    )))
}

fn recorder_error(error: impl Display) -> FractalError {
    FractalError::InvalidState(format!("checkpoint recorder failed: {error}"))
}

#[cfg(test)]
mod tests {
    use burn::{backend::Autodiff, backend::Candle, tensor::Int, Tensor};

    use super::*;
    use crate::{run_baseline_v2_smoke_train, V2SmokeTrainConfig};

    type TrainBackend = Autodiff<Candle<f32, i64>>;
    type TestBackend = Candle<f32, i64>;

    #[test]
    fn checkpoint_loader_rebuilds_trained_baseline_model_from_report() {
        let root = unique_temp_dir("v2-checkpoint-loader");
        let corpus_path = root.join("corpus.md");
        fs::write(&corpus_path, "checkpoint loader smoke corpus\n".repeat(32)).unwrap();
        let output_dir = root.join("artifacts");
        let mut config = V2SmokeTrainConfig::new(vec![corpus_path], output_dir.clone());
        config.train_steps = 2;
        config.eval_batches = 1;
        config.eval_holdout_every = 2;

        let train_device = Default::default();
        let result = run_baseline_v2_smoke_train::<TrainBackend>(config, &train_device).unwrap();

        let device = Default::default();
        let loaded = load_baseline_v2_checkpoint_model::<TestBackend>(
            &result.report.checkpoint.report_path,
            V2CheckpointSelection::Best,
            &device,
        )
        .unwrap();

        assert_eq!(loaded.selection, V2CheckpointSelection::Best);
        assert_eq!(loaded.report.config.model, result.report.config.model);
        assert_eq!(
            loaded.checkpoint_path,
            result.report.checkpoint.best_model_path
        );
        assert_eq!(
            loaded.model.shape().vocab_size,
            result.report.config.model.vocab_size
        );

        let input_ids = Tensor::<TestBackend, 2, Int>::zeros([1, 4], &device);
        let output = loaded.model.forward(input_ids).unwrap();
        assert_eq!(
            output.logits().dims(),
            [1, 4, result.report.config.model.vocab_size]
        );
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
