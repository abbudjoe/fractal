use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};

use crate::{error::FractalError, TournamentRunReport};

const DEFAULT_RESULTS_ROOT: &str = ".fractal-run-results";
const ARTIFACT_FILENAME: &str = "tournament-run-artifact.json";
const MANIFEST_FILENAME: &str = "tournament-run-manifest.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PersistedRunPaths {
    pub root: PathBuf,
    pub artifact_path: PathBuf,
    pub manifest_path: PathBuf,
}

pub fn persist_run_artifacts(
    report: &TournamentRunReport,
) -> Result<PersistedRunPaths, FractalError> {
    let paths = resolve_output_paths(report)?;
    if let Some(parent) = paths.artifact_path.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }
    if let Some(parent) = paths.manifest_path.parent() {
        fs::create_dir_all(parent).map_err(io_error)?;
    }

    fs::write(
        &paths.manifest_path,
        serde_json::to_vec_pretty(&build_manifest_json(report))
            .map_err(|error| FractalError::InvalidState(error.to_string()))?,
    )
    .map_err(io_error)?;
    fs::write(
        &paths.artifact_path,
        serde_json::to_vec_pretty(&build_artifact_json(report))
            .map_err(|error| FractalError::InvalidState(error.to_string()))?,
    )
    .map_err(io_error)?;

    Ok(paths)
}

fn resolve_output_paths(report: &TournamentRunReport) -> Result<PersistedRunPaths, FractalError> {
    let artifact_root = std::env::var_os("FRACTAL_RUN_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| default_run_root(report));
    let manifest_root = std::env::var_os("FRACTAL_RUN_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| artifact_root.clone());

    Ok(PersistedRunPaths {
        root: artifact_root.clone(),
        artifact_path: artifact_root.join(ARTIFACT_FILENAME),
        manifest_path: manifest_root.join(MANIFEST_FILENAME),
    })
}

fn default_run_root(report: &TournamentRunReport) -> PathBuf {
    let run_id = std::env::var("FRACTAL_RUN_ID").unwrap_or_else(|_| {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0);
        format!(
            "{}_{}_{}",
            millis,
            sanitize_path_component(report.preset.name()),
            sanitize_path_component(report.lane.name())
        )
    });
    Path::new(DEFAULT_RESULTS_ROOT).join(run_id)
}

fn build_manifest_json(report: &TournamentRunReport) -> Value {
    json!({
        "run_id": std::env::var("FRACTAL_RUN_ID").ok(),
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0),
        "commit_sha": detect_commit_sha(),
        "comparison_authority": report.comparison_label(),
        "preset": report.preset.name(),
        "lane": report.lane.name(),
        "backend": backend_name(&report.config.execution_backend),
        "execution_mode": execution_mode_name(report.config.execution_mode),
        "pod_id": std::env::var("FRACTAL_RUN_POD_ID").ok(),
        "timeout_seconds": report.config.run_timeout.map(|timeout| timeout.as_secs_f64()),
        "wrapper_timeout_seconds": std::env::var("FRACTAL_WRAPPER_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok()),
        "config": config_json(report),
    })
}

fn build_artifact_json(report: &TournamentRunReport) -> Value {
    let records = report
        .artifact
        .species
        .iter()
        .map(|record| {
            json!({
                "variant_name": record.manifest.variant_name.as_str(),
                "species": record.stage.species.as_str(),
                "ordinal": record.stage.ordinal,
                "total": record.stage.total,
                "outcome_class": outcome_class_name(record.outcome_class()),
                "execution_outcome": execution_outcome_name(record.execution_outcome),
                "quality_outcome": quality_outcome_name(record.quality_outcome),
                "error": record.error,
                "timeout_seconds": record.manifest.timeout_budget.map(|timeout| timeout.as_secs_f64()),
                "phase_timings": record.phase_timings.iter().map(|timing| {
                    json!({
                        "phase": phase_name(timing.phase),
                        "elapsed_seconds": timing.elapsed.as_secs_f64(),
                        "completed": timing.completed,
                        "total": timing.total,
                    })
                }).collect::<Vec<_>>(),
                "metrics": record.metrics.as_ref().map(|metrics| {
                    json!({
                        "grad_norm_depth_20": metrics.grad_norm_depth_20,
                        "long_context_perplexity": metrics.long_context_perplexity,
                        "arc_accuracy": metrics.arc_accuracy,
                        "tokens_per_sec": metrics.tokens_per_sec,
                    })
                }),
                "ranked_result": report.results.iter().find(|result| result.species == record.stage.species).map(|result| {
                    json!({
                        "rank": result.rank,
                        "fitness": result.fitness,
                        "stability_score": result.stability_score,
                        "long_context_perplexity": result.long_context_perplexity,
                        "arc_accuracy": result.arc_accuracy,
                        "tokens_per_sec": result.tokens_per_sec,
                    })
                }),
            })
        })
        .collect::<Vec<_>>();

    json!({
        "manifest": build_manifest_json(report),
        "results": records,
    })
}

fn config_json(report: &TournamentRunReport) -> Value {
    json!({
        "dim": report.config.dim,
        "levels": report.config.levels,
        "vocab_size": report.config.vocab_size,
        "max_seq_len": report.config.max_seq_len,
        "max_recursion_depth": report.config.max_recursion_depth,
        "stability_depth": report.config.stability_depth,
        "router_threshold": report.config.router_threshold,
        "train_batch_size": report.config.train_batch_size,
        "eval_batch_size": report.config.eval_batch_size,
        "train_steps_per_species": report.config.train_steps_per_species,
        "eval_batches_per_family": report.config.eval_batches_per_family,
        "learning_rate": report.config.learning_rate,
        "seed": report.config.seed,
        "parallelism": report.config.parallelism,
    })
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '.' | '_' | '-' => ch,
            _ => '_',
        })
        .collect()
}

fn detect_commit_sha() -> Option<String> {
    if let Ok(value) = std::env::var("FRACTAL_COMMIT_SHA") {
        if !value.is_empty() {
            return Some(value);
        }
    }

    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn backend_name(backend: &crate::ComputeBackend) -> &'static str {
    match backend {
        crate::ComputeBackend::CpuCandle => "cpu-candle",
        #[cfg(feature = "cuda")]
        crate::ComputeBackend::CudaCandle { .. } => "cuda-candle",
        crate::ComputeBackend::MetalWgpu { .. } => "metal-wgpu",
    }
}

fn execution_mode_name(mode: crate::ExecutionMode) -> &'static str {
    match mode {
        crate::ExecutionMode::Sequential => "sequential",
        crate::ExecutionMode::Parallel => "parallel",
    }
}

fn phase_name(phase: crate::RunPhase) -> &'static str {
    match phase {
        crate::RunPhase::Train => "train",
        crate::RunPhase::Stability => "stability",
        crate::RunPhase::Perplexity => "perplexity",
        crate::RunPhase::ArcSpeed => "arc_speed",
    }
}

fn execution_outcome_name(outcome: crate::RunExecutionOutcome) -> &'static str {
    match outcome {
        crate::RunExecutionOutcome::Success => "success",
        crate::RunExecutionOutcome::TrainTimeout => "train-timeout",
        crate::RunExecutionOutcome::EvalConstrained => "eval-constrained",
        crate::RunExecutionOutcome::InfraFailure => "infra-failure",
    }
}

fn quality_outcome_name(outcome: crate::RunQualityOutcome) -> &'static str {
    match outcome {
        crate::RunQualityOutcome::Clean => "clean",
        crate::RunQualityOutcome::NumericFailure => "numeric-failure",
        crate::RunQualityOutcome::LowSignal => "low-signal",
        crate::RunQualityOutcome::RuntimeCost => "runtime-cost",
    }
}

fn outcome_class_name(outcome: crate::RunOutcomeClass) -> &'static str {
    match outcome {
        crate::RunOutcomeClass::Success => "success",
        crate::RunOutcomeClass::TrainTimeout => "train-timeout",
        crate::RunOutcomeClass::EvalConstrained => "eval-constrained",
        crate::RunOutcomeClass::NumericFailure => "numeric-failure",
        crate::RunOutcomeClass::LowSignal => "low-signal",
        crate::RunOutcomeClass::RuntimeCost => "runtime-cost",
        crate::RunOutcomeClass::InfraFailure => "infra-failure",
    }
}

fn io_error(error: std::io::Error) -> FractalError {
    FractalError::InvalidState(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{env, fs};

    use crate::{
        species_registry_for_species, ComparisonAuthority, RankedSpeciesResult, SpeciesId,
        TournamentLane, TournamentPreset, TournamentRunReport,
    };

    use super::{persist_run_artifacts, ARTIFACT_FILENAME, MANIFEST_FILENAME};

    #[test]
    fn persist_run_artifacts_writes_manifest_and_artifact_json() {
        let temp_root =
            env::temp_dir().join(format!("fractal-run-artifacts-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        let artifact_dir = temp_root.join("artifacts");
        let manifest_dir = temp_root.join("manifests");
        env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let species = species_registry_for_species(SpeciesId::P1Contractive);
        let artifact = fractal_core::TournamentRunArtifact {
            config: TournamentPreset::FastTest.config(),
            species: vec![fractal_core::SpeciesRunArtifact {
                stage: fractal_core::SpeciesRunStage {
                    species: SpeciesId::P1Contractive,
                    ordinal: 1,
                    total: 1,
                },
                manifest: fractal_core::RunManifest {
                    variant_name: fractal_core::PrimitiveVariantName::new_unchecked(
                        "p1_contractive_v1",
                    ),
                    timeout_budget: None,
                    config: TournamentPreset::FastTest.config(),
                },
                phase_timings: vec![fractal_core::PhaseTiming {
                    phase: fractal_core::RunPhase::Train,
                    elapsed: std::time::Duration::from_secs(1),
                    completed: 1,
                    total: 1,
                }],
                execution_outcome: fractal_core::RunExecutionOutcome::Success,
                quality_outcome: fractal_core::RunQualityOutcome::Clean,
                error: None,
                metrics: Some(fractal_core::SpeciesRawMetrics {
                    species: SpeciesId::P1Contractive,
                    grad_norm_depth_20: 0.53,
                    long_context_perplexity: 1.54,
                    arc_accuracy: 0.68,
                    tokens_per_sec: 100.0,
                }),
            }],
        };
        let report = TournamentRunReport::new(
            TournamentPreset::FastTest,
            TournamentLane::Leader,
            ComparisonAuthority::AuthoritativeSamePreset,
            TournamentPreset::FastTest.config(),
            species,
            vec![RankedSpeciesResult {
                rank: 1,
                species: SpeciesId::P1Contractive,
                stability_score: 0.53,
                long_context_perplexity: 1.54,
                arc_accuracy: 0.68,
                tokens_per_sec: 100.0,
                fitness: 0.58,
            }],
            artifact,
        );

        let paths = persist_run_artifacts(&report).unwrap();
        assert!(paths.artifact_path.ends_with(ARTIFACT_FILENAME));
        assert!(paths.manifest_path.ends_with(MANIFEST_FILENAME));
        assert!(paths.artifact_path.exists());
        assert!(paths.manifest_path.exists());

        env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
        env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
        let _ = fs::remove_dir_all(&temp_root);
    }
}
