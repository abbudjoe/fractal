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
    let run_id = report_run_id(report)
        .or_else(|| std::env::var("FRACTAL_RUN_ID").ok())
        .unwrap_or_else(|| {
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
        "run_id": report_run_id(report).or_else(|| std::env::var("FRACTAL_RUN_ID").ok()),
        "generated_at_unix_ms": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis())
            .unwrap_or(0),
        "commit_sha": report_commit_sha(report).or_else(detect_commit_sha),
        "comparison_authority": report.comparison_label(),
        "comparison_contract": comparison_contract_json(&report.comparison),
        "runtime_surface_policy": report.runtime_surface_label(),
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
        "experiments": report
            .artifact
            .species
            .iter()
            .filter_map(|record| record.manifest.experiment.as_ref())
            .map(experiment_json)
            .collect::<Vec<_>>(),
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
                "comparison_authority": record
                    .manifest
                    .experiment
                    .as_ref()
                    .map(|experiment| experiment.comparison.label()),
                "runtime_surface_policy": record
                    .manifest
                    .experiment
                    .as_ref()
                    .map(|experiment| experiment.runtime.label()),
                "error": record.error,
                "timeout_seconds": record.manifest.timeout_budget.map(|timeout| timeout.as_secs_f64()),
                "experiment": record
                    .manifest
                    .experiment
                    .as_ref()
                    .map(experiment_json),
                "phase_timings": record.phase_timings.iter().map(|timing| {
                    json!({
                        "phase": phase_name(timing.phase),
                        "elapsed_seconds": timing.elapsed.as_secs_f64(),
                        "completed": timing.completed,
                        "total": timing.total,
                    })
                }).collect::<Vec<_>>(),
                "training_runtime": training_runtime_json(&record.training_runtime),
                "metrics": record.metrics.as_ref().map(|metrics| {
                    json!({
                        "grad_norm_depth_20": metrics.grad_norm_depth_20,
                        "long_context_perplexity": metrics.long_context_perplexity,
                        "arc_accuracy": metrics.arc_accuracy,
                        "tokens_per_sec": metrics.tokens_per_sec,
                    })
                }),
                "tokenizer_bridge": report.bridge_stats.get(&record.stage.species).map(tokenizer_bridge_json),
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
        "train_token_budget": report.config.train_token_budget,
        "eval_batches_per_family": report.config.eval_batches_per_family,
        "perplexity_eval_batches": report.config.perplexity_eval_batches,
        "arc_eval_batches": report.config.arc_eval_batches,
        "effective_perplexity_eval_batches": report.config.effective_perplexity_eval_batches(),
        "effective_arc_eval_batches": report.config.effective_arc_eval_batches(),
        "learning_rate": report.config.learning_rate,
        "optimizer": optimizer_json(&report.config.optimizer),
        "launch_policy": launch_policy_json(&report.config.launch_policy),
        "seed": report.config.seed,
        "parallelism": report.config.parallelism,
        "training_input": report
            .config
            .experiment
            .as_ref()
            .map(|experiment| training_input_json(&experiment.training_input)),
    })
}

fn comparison_contract_json(contract: &crate::ComparisonContract) -> Value {
    json!({
        "authority": match contract.authority {
            crate::ComparisonAuthority::Authoritative => "authoritative",
            crate::ComparisonAuthority::Advisory => "advisory",
        },
        "requires_same_preset": contract.requires_same_preset,
        "requires_same_runtime_surfaces": contract.requires_same_runtime_surfaces,
        "requires_frozen_commit": contract.requires_frozen_commit,
        "requires_same_backend": contract.requires_same_backend,
        "label": contract.label(),
    })
}

fn experiment_json(spec: &crate::ExperimentSpec) -> Value {
    json!({
        "experiment_id": {
            "logical_name": spec.experiment_id.logical_name,
            "run_id": spec.experiment_id.run_id,
            "branch": spec.experiment_id.branch,
            "commit_sha": spec.experiment_id.commit_sha,
            "created_at_unix_ms": spec.experiment_id.created_at_unix_ms,
        },
        "question": {
            "summary": spec.question.summary,
            "lane_intent": spec.question.lane_intent.as_str(),
            "decision_intent": spec.question.decision_intent.as_str(),
        },
        "variant": {
            "species": spec.variant.species.as_str(),
            "variant_name": spec.variant.variant_name.as_str(),
        },
        "model": {
            "architecture": spec.model.architecture.as_str(),
            "hidden_dim": spec.model.hidden_dim,
            "max_recursion_depth": spec.model.max_recursion_depth,
            "router_enabled": spec.model.router_enabled,
            "label": spec.model.label(),
        },
        "optimizer": optimizer_json(&spec.optimizer),
        "training_input": training_input_json(&spec.training_input),
        "budget": {
            "preset": spec.budget.preset.name(),
            "seed": spec.budget.seed,
            "train_batch_size": spec.budget.train_batch_size,
            "eval_batch_size": spec.budget.eval_batch_size,
            "train_steps_per_species": spec.budget.train_steps_per_species,
            "train_token_budget": spec.budget.train_token_budget,
            "eval_batches_per_family": spec.budget.eval_batches_per_family,
            "perplexity_eval_batches": spec.budget.perplexity_eval_batches,
            "arc_eval_batches": spec.budget.arc_eval_batches,
            "max_recursion_depth": spec.budget.max_recursion_depth,
            "stability_depth": spec.budget.stability_depth,
            "learning_rate": spec.budget.learning_rate,
            "timeout_seconds": spec.budget.timeout_seconds,
        },
        "runtime": {
            "eval_backend_policy": spec.runtime.eval_backend_policy.as_str(),
            "batching_policy": spec.runtime.batching_policy.as_str(),
            "execution_policy": spec.runtime.execution_policy.as_str(),
            "buffer_reuse_policy": spec.runtime.buffer_reuse_policy.as_str(),
            "benchmark_mode": spec.runtime.benchmark_mode.as_str(),
            "backend_policy": spec.runtime.backend_policy.as_str(),
            "launch_policy": launch_policy_json(&spec.runtime.launch_policy),
            "label": spec.runtime.label(),
        },
        "comparison": comparison_contract_json(&spec.comparison),
        "execution": {
            "kind": spec.execution.kind.as_str(),
            "backend": spec.execution.backend.as_str(),
            "execution_mode": execution_mode_name(spec.execution.execution_mode),
            "pod_id": spec.execution.pod_id,
            "wrapper_timeout_seconds": spec.execution.wrapper_timeout_seconds,
        },
        "artifacts": {
            "manifest_required": spec.artifacts.manifest_required,
            "structured_artifact_required": spec.artifacts.structured_artifact_required,
            "final_log_required": spec.artifacts.final_log_required,
            "tracker_ready_output_required": spec.artifacts.tracker_ready_output_required,
        },
    })
}

fn optimizer_json(spec: &crate::OptimizerSpec) -> Value {
    json!({
        "kind": spec.kind.as_str(),
        "peak_learning_rate": spec.peak_learning_rate,
        "beta_1": spec.beta_1,
        "beta_2": spec.beta_2,
        "epsilon": spec.epsilon,
        "weight_decay": spec.weight_decay,
        "gradient_clip_norm": spec.gradient_clip_norm,
        "schedule": {
            "kind": spec.schedule.kind.as_str(),
            "warmup_fraction": spec.schedule.warmup_fraction,
            "decay_floor_fraction": spec.schedule.decay_floor_fraction,
            "label": spec.schedule.label(),
        },
        "label": spec.label(),
    })
}

fn training_input_json(spec: &crate::TrainingInputSpec) -> Value {
    json!({
        "mode": spec.mode.as_str(),
        "corpus_name": spec.corpus_name,
        "corpus_source": spec.corpus_source.as_ref().map(|source| {
            json!({
                "train": text_corpus_split_json(&source.train),
                "eval": text_corpus_split_json(&source.eval),
            })
        }),
        "arc_source": {
            "mode": spec.arc_source.mode.as_str(),
        },
        "bridge": {
            "enabled": spec.bridge.enabled,
            "observational_only": spec.bridge.observational_only,
        },
        "tokenizer": spec.tokenizer.as_ref().map(|tokenizer| {
            json!({
                "artifact_id": tokenizer.artifact_id,
                "artifact_path": tokenizer.artifact_path,
                "vocab_size": tokenizer.vocab_size,
                "pad_token_id": tokenizer.pad_token_id,
            })
        }),
        "bridge_packaging": spec.bridge_packaging.as_ref().map(|bridge_packaging| {
            json!({
                "vocab_artifact_path": bridge_packaging.vocab_artifact_path,
                "dim": bridge_packaging.dim,
                "levels": bridge_packaging.levels,
                "max_depth": bridge_packaging.max_depth,
                "seed": bridge_packaging.seed,
                "split_policy": bridge_packaging.split_policy.as_str(),
                "substrate_mode": bridge_packaging.substrate_mode.as_str(),
                "chunk_max_tokens": bridge_packaging.chunk_max_tokens,
                "chunk_max_bytes": bridge_packaging.chunk_max_bytes,
            })
        }),
    })
}

fn training_runtime_json(spec: &fractal_core::lifecycle::TrainingRuntimeArtifact) -> Value {
    json!({
        "completed_steps": spec.completed_steps,
        "planned_steps": spec.planned_steps,
        "train_tokens_seen": spec.train_tokens_seen,
        "target_train_tokens": spec.target_train_tokens,
        "resumed_from_checkpoint": spec.resumed_from_checkpoint,
        "checkpoints": spec.checkpoints.iter().map(|checkpoint| {
            json!({
                "kind": checkpoint.kind.as_str(),
                "tokens_seen": checkpoint.tokens_seen,
                "completed_steps": checkpoint.completed_steps,
                "directory": checkpoint.directory,
                "long_context_perplexity": checkpoint.long_context_perplexity,
            })
        }).collect::<Vec<_>>(),
        "weight_export": weight_export_runtime_state_json(&spec.weight_export),
        "failure_snapshot": failure_snapshot_runtime_state_json(&spec.failure_snapshot),
        "interim_evaluations": spec.interim_evaluations.iter().map(|snapshot| {
            json!({
                "tokens_seen": snapshot.tokens_seen,
                "completed_steps": snapshot.completed_steps,
                "stability_score": snapshot.stability_score,
                "long_context_perplexity": snapshot.long_context_perplexity,
                "arc_accuracy": snapshot.arc_accuracy,
                "tokens_per_sec": snapshot.tokens_per_sec,
            })
        }).collect::<Vec<_>>(),
        "diagnostics": diagnostics_runtime_json(&spec.diagnostics),
    })
}

fn weight_export_runtime_state_json(spec: &crate::WeightExportRuntimeState) -> Value {
    json!({
        "policy": {
            "format": weight_export_format_json(&spec.policy.format),
            "phases": spec
                .policy
                .phases
                .iter()
                .map(|phase| phase_name_from_weight_export(*phase))
                .collect::<Vec<_>>(),
            "required": spec.policy.required,
            "label": spec.policy.label(),
        },
        "completeness": spec.completeness.as_str(),
        "completed_artifacts": spec
            .completed_artifacts()
            .map(weight_export_artifact_json)
            .collect::<Vec<_>>(),
        "attempts": spec
            .attempts
            .iter()
            .map(weight_export_attempt_json)
            .collect::<Vec<_>>(),
        "missing_required_phases": spec
            .missing_required_phases
            .iter()
            .map(|phase| phase_name_from_weight_export(*phase))
            .collect::<Vec<_>>(),
    })
}

fn weight_export_attempt_json(spec: &crate::WeightExportAttempt) -> Value {
    match &spec.outcome {
        crate::WeightExportAttemptOutcome::Succeeded { artifact } => json!({
            "phase": phase_name_from_weight_export(spec.phase),
            "status": "succeeded",
            "artifact": weight_export_artifact_json(artifact),
        }),
        crate::WeightExportAttemptOutcome::Failed { error } => json!({
            "phase": phase_name_from_weight_export(spec.phase),
            "status": "failed",
            "error": error,
        }),
    }
}

fn weight_export_artifact_json(spec: &crate::WeightExportArtifact) -> Value {
    json!({
        "format": weight_export_format_json(&spec.format),
        "phase": phase_name_from_weight_export(spec.phase),
        "path": spec.path,
        "metadata_path": spec.metadata_path,
        "required": spec.required,
        "contract": weight_export_contract_json(&spec.contract),
    })
}

fn weight_export_contract_json(spec: &crate::WeightExportContract) -> Value {
    json!({
        "experiment_logical_name": spec.experiment_logical_name,
        "experiment_run_id": spec.experiment_run_id,
        "experiment_branch": spec.experiment_branch,
        "experiment_commit_sha": spec.experiment_commit_sha,
        "species": spec.species,
        "variant_name": spec.variant_name,
        "model": {
            "architecture": spec.model.architecture.as_str(),
            "hidden_dim": spec.model.hidden_dim,
            "max_recursion_depth": spec.model.max_recursion_depth,
            "router_enabled": spec.model.router_enabled,
            "label": spec.model.label(),
        },
        "vocab_size": spec.vocab_size,
        "precision": {
            "compute": spec.precision.compute.as_str(),
            "optimizer_state": spec.precision.optimizer_state.as_str(),
            "reduction": spec.precision.reduction.as_str(),
            "tf32_enabled": spec.precision.tf32_enabled,
            "quantization": {
                "weights": spec.precision.quantization.weights.map(|kind| kind.as_str()),
                "activations": spec.precision.quantization.activations.map(|kind| kind.as_str()),
            },
        },
        "format": weight_export_format_json(&spec.format),
    })
}

fn weight_export_format_json(format: &crate::WeightExportFormat) -> Value {
    match format {
        crate::WeightExportFormat::BurnBin => json!({
            "format": "burn-bin",
        }),
        crate::WeightExportFormat::SafeTensors => json!({
            "format": "safe-tensors",
        }),
        crate::WeightExportFormat::Quantized { precision } => json!({
            "format": "quantized",
            "precision": precision.as_str(),
        }),
    }
}

fn phase_name_from_weight_export(phase: crate::WeightExportPhase) -> &'static str {
    match phase {
        crate::WeightExportPhase::Best => "best",
        crate::WeightExportPhase::Final => "final",
        crate::WeightExportPhase::Latest => "latest",
        crate::WeightExportPhase::FailureSnapshot => "failure-snapshot",
    }
}

fn failure_snapshot_runtime_state_json(spec: &crate::FailureSnapshotRuntimeState) -> Value {
    json!({
        "policy": {
            "enabled": spec.policy.enabled,
            "required": spec.policy.required,
            "capture_model_weights": spec.policy.capture_model_weights,
            "capture_runtime_state": spec.policy.capture_runtime_state,
            "capture_diagnostics_tail": spec.policy.capture_diagnostics_tail,
            "label": spec.policy.label(),
        },
        "attempted": spec.attempted,
        "completeness": spec.completeness.as_str(),
        "contract": spec.contract.as_ref().map(failure_snapshot_contract_json),
        "artifacts": spec
            .captured_artifacts()
            .map(failure_snapshot_artifact_json)
            .collect::<Vec<_>>(),
        "attempts": spec
            .attempts
            .iter()
            .map(failure_snapshot_attempt_json)
            .collect::<Vec<_>>(),
        "missing_required_artifacts": spec
            .missing_required_artifacts
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>(),
        "last_diagnostic_event": spec
            .last_diagnostic_event
            .as_ref()
            .map(diagnostic_event_summary_json),
        "last_rule_projection_event": spec
            .last_rule_projection_event
            .as_ref()
            .map(rule_projection_diagnostic_event_summary_json),
    })
}

fn failure_snapshot_contract_json(spec: &crate::FailureSnapshotContract) -> Value {
    json!({
        "experiment_logical_name": spec.experiment_logical_name,
        "experiment_run_id": spec.experiment_run_id,
        "experiment_branch": spec.experiment_branch,
        "experiment_commit_sha": spec.experiment_commit_sha,
        "species": spec.species,
        "variant_name": spec.variant_name,
        "model": {
            "architecture": spec.model.architecture.as_str(),
            "hidden_dim": spec.model.hidden_dim,
            "max_recursion_depth": spec.model.max_recursion_depth,
            "router_enabled": spec.model.router_enabled,
            "label": spec.model.label(),
        },
        "vocab_size": spec.vocab_size,
        "precision": {
            "compute": spec.precision.compute.as_str(),
            "optimizer_state": spec.precision.optimizer_state.as_str(),
            "reduction": spec.precision.reduction.as_str(),
            "tf32_enabled": spec.precision.tf32_enabled,
            "quantization": {
                "weights": spec.precision.quantization.weights.map(|kind| kind.as_str()),
                "activations": spec.precision.quantization.activations.map(|kind| kind.as_str()),
            },
        },
        "error_class": spec.error_class.as_str(),
        "capture_timing": spec.capture_timing.as_str(),
        "last_successful_boundary": spec
            .last_successful_boundary
            .map(|boundary| boundary.as_str()),
    })
}

fn failure_snapshot_artifact_json(spec: &crate::FailureSnapshotArtifact) -> Value {
    json!({
        "kind": spec.kind.as_str(),
        "format": failure_snapshot_artifact_format_json(&spec.format),
        "path": spec.path,
    })
}

fn failure_snapshot_artifact_format_json(spec: &crate::FailureSnapshotArtifactFormat) -> Value {
    match spec {
        crate::FailureSnapshotArtifactFormat::Json => json!({
            "kind": "json",
        }),
        crate::FailureSnapshotArtifactFormat::BurnBin => json!({
            "kind": "burn-bin",
        }),
        crate::FailureSnapshotArtifactFormat::SafeTensors => json!({
            "kind": "safe-tensors",
        }),
        crate::FailureSnapshotArtifactFormat::Quantized { precision } => json!({
            "kind": "quantized",
            "precision": precision.as_str(),
        }),
    }
}

fn failure_snapshot_attempt_json(spec: &crate::FailureSnapshotAttempt) -> Value {
    match &spec.outcome {
        crate::FailureSnapshotAttemptOutcome::Captured { artifact } => json!({
            "kind": spec.kind.as_str(),
            "status": "captured",
            "artifact": failure_snapshot_artifact_json(artifact),
        }),
        crate::FailureSnapshotAttemptOutcome::Failed { error } => json!({
            "kind": spec.kind.as_str(),
            "status": "failed",
            "error": error,
        }),
    }
}

fn text_corpus_split_json(spec: &crate::TextCorpusSplitSpec) -> Value {
    json!({
        "path": spec.path,
        "format": match &spec.format {
            crate::TextCorpusFormat::JsonlText { text_field } => json!({
                "format": "jsonl-text",
                "text_field": text_field,
            }),
            crate::TextCorpusFormat::PlainTextLines => json!({
                "format": "plain-text-lines",
            }),
        },
        "max_documents": spec.max_documents,
    })
}

fn launch_policy_json(spec: &crate::LaunchPolicySpec) -> Value {
    json!({
        "label": spec.label(),
        "precision": {
            "compute": spec.precision.compute.as_str(),
            "optimizer_state": spec.precision.optimizer_state.as_str(),
            "reduction": spec.precision.reduction.as_str(),
            "tf32_enabled": spec.precision.tf32_enabled,
            "quantization": {
                "weights": spec.precision.quantization.weights.map(|kind| kind.as_str()),
                "activations": spec.precision.quantization.activations.map(|kind| kind.as_str()),
            },
        },
        "checkpoint": {
            "interval_tokens": spec.checkpoint.interval_tokens,
            "keep_latest": spec.checkpoint.keep_latest,
            "keep_best": spec.checkpoint.keep_best,
            "keep_final": spec.checkpoint.keep_final,
            "keep_previous": spec.checkpoint.keep_previous,
        },
        "eval_cadence": {
            "perplexity_interval_tokens": spec.eval_cadence.perplexity_interval_tokens,
            "stability_interval_tokens": spec.eval_cadence.stability_interval_tokens,
            "arc_interval_tokens": spec.eval_cadence.arc_interval_tokens,
            "systems_speed_interval_tokens": spec.eval_cadence.systems_speed_interval_tokens,
            "final_full_eval": spec.eval_cadence.final_full_eval,
        },
        "resume": {
            "resume_on_interrupt": spec.resume.resume_on_interrupt,
            "restart_on_corruption": spec.resume.restart_on_corruption,
            "restart_on_contract_ambiguity": spec.resume.restart_on_contract_ambiguity,
        },
        "weight_export": {
            "format": weight_export_format_json(&spec.weight_export.format),
            "phases": spec
                .weight_export
                .phases
                .iter()
                .map(|phase| phase_name_from_weight_export(*phase))
                .collect::<Vec<_>>(),
            "required": spec.weight_export.required,
            "label": spec.weight_export.label(),
        },
        "diagnostics": diagnostics_policy_json(&spec.diagnostics),
        "failure_snapshot": {
            "enabled": spec.failure_snapshot.enabled,
            "required": spec.failure_snapshot.required,
            "capture_model_weights": spec.failure_snapshot.capture_model_weights,
            "capture_runtime_state": spec.failure_snapshot.capture_runtime_state,
            "capture_diagnostics_tail": spec.failure_snapshot.capture_diagnostics_tail,
            "label": spec.failure_snapshot.label(),
        },
    })
}

fn diagnostics_policy_json(spec: &crate::DiagnosticsPolicy) -> Value {
    json!({
        "required": spec.required,
        "structured_output": spec.structured_output.as_str(),
        "probes": spec.probes.iter().map(diagnostic_probe_json).collect::<Vec<_>>(),
        "label": spec.label(),
    })
}

fn diagnostic_probe_json(spec: &crate::DiagnosticProbeRequest) -> Value {
    json!({
        "kind": spec.kind.as_str(),
        "cadence": probe_cadence_json(&spec.cadence),
        "position_interval": spec.position_interval,
    })
}

fn probe_cadence_json(spec: &crate::ProbeCadence) -> Value {
    match spec {
        crate::ProbeCadence::EveryStep => json!({
            "kind": "every_step",
        }),
        crate::ProbeCadence::StepInterval { steps } => json!({
            "kind": "step_interval",
            "steps": steps,
        }),
        crate::ProbeCadence::FirstNSteps { steps } => json!({
            "kind": "first_n_steps",
            "steps": steps,
        }),
    }
}

fn diagnostics_runtime_json(spec: &crate::DiagnosticsRuntimeArtifact) -> Value {
    json!({
        "policy": diagnostics_policy_json(&spec.policy),
        "structured_output": {
            "format": spec.policy.structured_output.as_str(),
            "event_file": spec.event_file,
        },
        "event_count": spec.events.len(),
        "emitted_probe_kinds": spec
            .emitted_probe_kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>(),
        "missing_required_probe_kinds": spec
            .missing_required_probe_kinds
            .iter()
            .map(|kind| kind.as_str())
            .collect::<Vec<_>>(),
        "missing_required_boundary_completions": spec
            .missing_required_boundary_completions
            .iter()
            .map(|boundary| boundary.as_str())
            .collect::<Vec<_>>(),
        "runtime_failure": spec
            .runtime_failure
            .as_ref()
            .map(diagnostics_runtime_failure_json),
        "diagnostics_incomplete": spec.diagnostics_incomplete,
        "boundary_memory_deltas": spec
            .boundary_memory_deltas
            .iter()
            .map(boundary_memory_delta_json)
            .collect::<Vec<_>>(),
        "last_event": spec.last_event.as_ref().map(diagnostic_event_json),
        "last_rule_projection_event": spec
            .last_rule_projection_event
            .as_ref()
            .map(rule_projection_diagnostic_event_summary_json),
    })
}

fn diagnostics_runtime_failure_json(spec: &crate::DiagnosticsRuntimeFailure) -> Value {
    json!({
        "kind": match spec.kind {
            crate::DiagnosticsRuntimeFailureKind::Initialization => "initialization",
            crate::DiagnosticsRuntimeFailureKind::EventPersistence => "event_persistence",
        },
        "probe_kind": spec.probe_kind.map(|kind| kind.as_str()),
        "boundary": spec.boundary.map(|boundary| boundary.as_str()),
        "message": spec.message.clone(),
    })
}

fn diagnostic_event_json(spec: &crate::DiagnosticEvent) -> Value {
    json!({
        "experiment_run_id": spec.experiment_run_id.clone(),
        "experiment_logical_name": spec.experiment_logical_name.clone(),
        "species": spec.species.clone(),
        "variant_name": spec.variant_name.clone(),
        "correlation_id": spec.correlation_id,
        "phase": phase_name(spec.phase),
        "step": spec.step,
        "tokens_seen": spec.tokens_seen,
        "event": diagnostic_event_kind_json(&spec.event),
    })
}

fn diagnostic_event_kind_json(spec: &crate::DiagnosticEventKind) -> Value {
    match spec {
        crate::DiagnosticEventKind::TrainStepStart {
            planned_steps,
            batch_token_count,
            input_shape,
            target_shape,
        } => json!({
            "kind": "train_step_start",
            "planned_steps": planned_steps,
            "batch_token_count": batch_token_count,
            "input_shape": input_shape,
            "target_shape": target_shape,
        }),
        crate::DiagnosticEventKind::ForwardStart { input_shape } => json!({
            "kind": "forward_start",
            "input_shape": input_shape,
        }),
        crate::DiagnosticEventKind::ForwardPosition {
            position,
            sequence_length,
            input_shape,
            readout_shape,
        } => json!({
            "kind": "forward_position",
            "position": position,
            "sequence_length": sequence_length,
            "input_shape": input_shape,
            "readout_shape": readout_shape,
        }),
        crate::DiagnosticEventKind::RuleProjection {
            position,
            sequence_length,
            recursion_depth,
            max_recursion_depth,
            projection,
            spec,
        } => json!({
            "kind": "rule_projection",
            "position": position,
            "sequence_length": sequence_length,
            "recursion_depth": recursion_depth,
            "max_recursion_depth": max_recursion_depth,
            "projection": projection.as_str(),
            "identity": rule_projection_identity_json(&spec.identity),
            "input_layout": tensor_layout_metadata_json(&spec.input_layout),
            "output_layout": tensor_layout_metadata_json(&spec.output_layout),
            "linear_layout": linear_projection_layout_metadata_json(spec.linear_layout.as_ref()),
        }),
        crate::DiagnosticEventKind::ForwardComplete {
            logits_shape,
            graph_burden,
        } => json!({
            "kind": "forward_complete",
            "logits_shape": logits_shape,
            "graph_burden": forward_graph_burden_json(graph_burden),
        }),
        crate::DiagnosticEventKind::LossStart { target_shape } => json!({
            "kind": "loss_start",
            "target_shape": target_shape,
        }),
        crate::DiagnosticEventKind::LossComplete { loss_shape } => json!({
            "kind": "loss_complete",
            "loss_shape": loss_shape,
        }),
        crate::DiagnosticEventKind::BackwardStart => json!({
            "kind": "backward_start",
        }),
        crate::DiagnosticEventKind::BackwardComplete => json!({
            "kind": "backward_complete",
        }),
        crate::DiagnosticEventKind::OptimizerStepStart {
            scheduled_learning_rate,
        } => json!({
            "kind": "optimizer_step_start",
            "scheduled_learning_rate": scheduled_learning_rate,
        }),
        crate::DiagnosticEventKind::OptimizerStepComplete {
            scheduled_learning_rate,
            completed_steps,
            train_tokens_seen,
        } => json!({
            "kind": "optimizer_step_complete",
            "scheduled_learning_rate": scheduled_learning_rate,
            "completed_steps": completed_steps,
            "train_tokens_seen": train_tokens_seen,
        }),
        crate::DiagnosticEventKind::CudaMemorySnapshot {
            boundary,
            used_mib,
            free_mib,
            total_mib,
        } => json!({
            "kind": "cuda_memory_snapshot",
            "boundary": boundary.as_str(),
            "used_mib": used_mib,
            "free_mib": free_mib,
            "total_mib": total_mib,
        }),
    }
}

fn diagnostic_event_summary_json(spec: &crate::DiagnosticEventSummary) -> Value {
    json!({
        "correlation_id": spec.correlation_id,
        "phase": phase_name(spec.phase),
        "step": spec.step,
        "tokens_seen": spec.tokens_seen,
        "event_name": spec.event_name,
    })
}

fn rule_projection_diagnostic_event_summary_json(
    spec: &crate::RuleProjectionDiagnosticEventSummary,
) -> Value {
    json!({
        "correlation_id": spec.correlation_id,
        "phase": phase_name(spec.phase),
        "step": spec.step,
        "tokens_seen": spec.tokens_seen,
        "position": spec.position,
        "sequence_length": spec.sequence_length,
        "recursion_depth": spec.recursion_depth,
        "max_recursion_depth": spec.max_recursion_depth,
        "projection": spec.projection.as_str(),
        "identity": rule_projection_identity_json(&spec.identity),
        "input_layout": tensor_layout_metadata_json(&spec.input_layout),
        "output_layout": tensor_layout_metadata_json(&spec.output_layout),
        "linear_layout": linear_projection_layout_metadata_json(spec.linear_layout.as_ref()),
    })
}

fn rule_projection_identity_json(spec: &crate::RuleProjectionIdentity) -> Value {
    json!({
        "rule_name": spec.rule_name,
        "projection_name": spec.projection_name,
    })
}

fn tensor_layout_metadata_json(spec: &crate::TensorLayoutMetadata) -> Value {
    json!({
        "origin": match spec.origin {
            crate::TensorLayoutOrigin::RuntimeObserved => "runtime_observed",
            crate::TensorLayoutOrigin::LinearContract => "linear_contract",
        },
        "transform": match spec.transform {
            crate::TensorLayoutTransform::Identity => "identity",
            crate::TensorLayoutTransform::UnsqueezedView => "unsqueezed_view",
            crate::TensorLayoutTransform::TransposedView => "transposed_view",
        },
        "shape": spec.shape,
        "strides": spec.strides,
        "is_contiguous": spec.is_contiguous,
    })
}

fn linear_projection_layout_metadata_json(
    spec: Option<&crate::LinearProjectionLayoutMetadata>,
) -> Value {
    match spec {
        Some(spec) => json!({
            "stored_weight": tensor_layout_metadata_json(&spec.stored_weight),
            "forward_rhs": tensor_layout_metadata_json(&spec.forward_rhs),
            "backward_input_grad_rhs": tensor_layout_metadata_json(&spec.backward_input_grad_rhs),
        }),
        None => Value::Null,
    }
}

fn forward_graph_burden_json(spec: &crate::ForwardGraphBurden) -> Value {
    json!({
        "batch_size": spec.batch_size,
        "sequence_length": spec.sequence_length,
        "positions_processed": spec.positions_processed,
        "rule_invocations": spec.rule_invocations,
        "max_recursion_depth_observed": spec.max_recursion_depth_observed,
        "router_enabled": spec.router_enabled,
        "forced_depth": spec.forced_depth,
        "positions_early_exited": spec.positions_early_exited,
        "positions_reached_depth_limit": spec.positions_reached_depth_limit,
        "router_exit_elements": spec.router_exit_elements,
        "router_continue_elements": spec.router_continue_elements,
    })
}

fn boundary_memory_delta_json(spec: &crate::BoundaryMemoryDelta) -> Value {
    json!({
        "boundary": spec.boundary.as_str(),
        "step": spec.step,
        "tokens_seen": spec.tokens_seen,
        "correlation_id": spec.correlation_id,
        "used_mib": spec.used_mib,
        "free_mib": spec.free_mib,
        "total_mib": spec.total_mib,
        "delta_used_mib": spec.delta_used_mib,
        "delta_free_mib": spec.delta_free_mib,
        "delta_total_mib": spec.delta_total_mib,
    })
}

fn tokenizer_bridge_json(stats: &crate::TokenizerBridgeStats) -> Value {
    json!({
        "corpus_name": stats.corpus_name,
        "tokenizer_artifact_id": stats.tokenizer_artifact_id,
        "bridge_vocab_artifact_path": stats.bridge_vocab_artifact_path,
        "bridge_split_policy": stats.bridge_split_policy.as_str(),
        "bridge_substrate_mode": stats.bridge_substrate_mode.as_str(),
        "training_input_mode": stats.training_input_mode.as_str(),
        "bridge_enabled": stats.bridge_enabled,
        "bridge_observational_only": stats.bridge_observational_only,
        "arc_source_mode": stats.arc_source_mode.as_str(),
        "native_pad_token_id": stats.native_pad_token_id,
        "canonical_model_pad_token_id": stats.canonical_model_pad_token_id,
        "uses_canonical_pad_alias": stats.uses_canonical_pad_alias,
        "train_documents": stats.train_documents,
        "eval_documents": stats.eval_documents,
        "model_facing_documents": stats.model_facing_documents,
        "bridge_documents": stats.bridge_documents,
        "bridge_chunks": stats.bridge_chunks,
        "bridge_tokens": stats.bridge_tokens,
        "native_documents": stats.native_documents,
        "native_chunks": stats.native_chunks,
        "native_tokens": stats.native_tokens,
        "train_batches": stats.train_batches,
        "eval_batches": stats.eval_batches,
        "sequence_len": stats.sequence_len,
    })
}

fn report_run_id(report: &TournamentRunReport) -> Option<String> {
    report
        .artifact
        .species
        .first()
        .and_then(|record| record.manifest.experiment.as_ref())
        .map(|experiment| experiment.experiment_id.run_id.clone())
}

fn report_commit_sha(report: &TournamentRunReport) -> Option<String> {
    report
        .artifact
        .species
        .first()
        .and_then(|record| record.manifest.experiment.as_ref())
        .and_then(|experiment| experiment.experiment_id.commit_sha.clone())
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
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::{env, fs};

    use crate::{
        artifact_env_test_lock, species_registry_for_species, ComparisonContract, DiagnosticEvent,
        DiagnosticEventKind, DiagnosticProbeKind, DiagnosticProbeRequest, DiagnosticsPolicy,
        RankedSpeciesResult, SpeciesId, TournamentLane, TournamentPreset, TournamentRunReport,
        TournamentRunReportParts,
    };
    use fractal_core::lifecycle::TrainingRuntimeArtifact;
    use fractal_core::{
        ArtifactPolicy, BudgetSpec, DecisionIntent, ExecutionBackend, ExecutionTarget,
        ExecutionTargetKind, ExperimentId, ExperimentQuestion, ExperimentSpec, LaneIntent,
        OptimizerSpec, ProbeCadence, RunPhase, RuntimeSurfaceSpec, TrainingInputSpec, VariantSpec,
        WeightExportArtifact, WeightExportContract, WeightExportFormat, WeightExportPhase,
        WeightExportPolicy,
    };

    use super::{persist_run_artifacts, ARTIFACT_FILENAME, MANIFEST_FILENAME};

    #[test]
    fn persist_run_artifacts_writes_manifest_and_artifact_json() {
        let _env_lock = artifact_env_test_lock();
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
                    experiment: Some(test_experiment_spec()),
                },
                phase_timings: vec![fractal_core::PhaseTiming {
                    phase: fractal_core::RunPhase::Train,
                    elapsed: std::time::Duration::from_secs(1),
                    completed: 1,
                    total: 1,
                }],
                training_runtime: TrainingRuntimeArtifact::default(),
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
        let report = TournamentRunReport::new(TournamentRunReportParts {
            preset: TournamentPreset::FastTest,
            lane: TournamentLane::Leader,
            comparison: ComparisonContract::authoritative_same_preset(),
            config: TournamentPreset::FastTest.config(),
            species,
            results: vec![RankedSpeciesResult {
                rank: 1,
                species: SpeciesId::P1Contractive,
                stability_score: 0.53,
                long_context_perplexity: 1.54,
                arc_accuracy: 0.68,
                tokens_per_sec: 100.0,
                fitness: 0.58,
            }],
            artifact,
            bridge_stats: BTreeMap::new(),
        });

        let paths = persist_run_artifacts(&report).unwrap();
        assert!(paths.artifact_path.ends_with(ARTIFACT_FILENAME));
        assert!(paths.manifest_path.ends_with(MANIFEST_FILENAME));
        assert!(paths.artifact_path.exists());
        assert!(paths.manifest_path.exists());

        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.manifest_path).unwrap()).unwrap();
        let artifact_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.artifact_path).unwrap()).unwrap();
        assert_eq!(
            manifest["comparison_contract"]["authority"],
            serde_json::Value::String("authoritative".to_owned())
        );
        assert_eq!(
            manifest["runtime_surface_policy"],
            serde_json::Value::String("conservative-defaults".to_owned())
        );
        assert_eq!(
            manifest["config"]["optimizer"]["kind"],
            serde_json::Value::String("adam".to_owned())
        );
        assert_eq!(
            manifest["config"]["launch_policy"]["precision"]["compute"],
            serde_json::Value::String("backend-default".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["comparison_authority"],
            serde_json::Value::String("authoritative same-preset".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["runtime_surface_policy"],
            serde_json::Value::String("conservative-defaults".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["experiment"]["variant"]["variant_name"],
            serde_json::Value::String("p1_contractive_v1".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["experiment"]["optimizer"]["kind"],
            serde_json::Value::String("adam".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["experiment"]["runtime"]["launch_policy"]["resume"]
                ["resume_on_interrupt"],
            serde_json::Value::Bool(false)
        );

        env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
        env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
        let _ = fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn persist_run_artifacts_references_runtime_owned_diagnostic_event_stream_and_summary() {
        let _env_lock = artifact_env_test_lock();
        let temp_root =
            env::temp_dir().join(format!("fractal-run-diagnostics-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        let artifact_dir = temp_root.join("artifacts");
        let manifest_dir = temp_root.join("manifests");
        let diagnostic_event_path = artifact_dir.join("diagnostics/run-123/p1_contractive.jsonl");
        fs::create_dir_all(diagnostic_event_path.parent().unwrap()).unwrap();
        fs::write(
            &diagnostic_event_path,
            "{\"event\":{\"kind\":\"train_step_start\"}}\n",
        )
        .unwrap();
        env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let species = species_registry_for_species(SpeciesId::P1Contractive);
        let training_runtime = TrainingRuntimeArtifact {
            diagnostics: fractal_core::DiagnosticsRuntimeArtifact {
                policy: DiagnosticsPolicy {
                    required: true,
                    probes: vec![DiagnosticProbeRequest {
                        kind: DiagnosticProbeKind::TrainStep,
                        cadence: ProbeCadence::EveryStep,
                        position_interval: None,
                    }],
                    structured_output: fractal_core::StructuredDiagnosticsOutput::Jsonl,
                },
                event_file: Some(diagnostic_event_path.display().to_string()),
                events: vec![DiagnosticEvent {
                    experiment_run_id: "run-123".to_owned(),
                    experiment_logical_name: Some("fast-test-run".to_owned()),
                    species: SpeciesId::P1Contractive.as_str().to_owned(),
                    variant_name: "p1_contractive_v1".to_owned(),
                    correlation_id: 1,
                    phase: RunPhase::Train,
                    step: Some(0),
                    tokens_seen: Some(0),
                    event: DiagnosticEventKind::TrainStepStart {
                        planned_steps: 1,
                        batch_token_count: 16,
                        input_shape: vec![1, 16],
                        target_shape: vec![1, 16],
                    },
                }],
                emitted_probe_kinds: vec![DiagnosticProbeKind::TrainStep],
                missing_required_probe_kinds: Vec::new(),
                missing_required_boundary_completions: Vec::new(),
                runtime_failure: None,
                diagnostics_incomplete: false,
                boundary_memory_deltas: Vec::new(),
                last_event: Some(DiagnosticEvent {
                    experiment_run_id: "run-123".to_owned(),
                    experiment_logical_name: Some("fast-test-run".to_owned()),
                    species: SpeciesId::P1Contractive.as_str().to_owned(),
                    variant_name: "p1_contractive_v1".to_owned(),
                    correlation_id: 1,
                    phase: RunPhase::Train,
                    step: Some(0),
                    tokens_seen: Some(0),
                    event: DiagnosticEventKind::TrainStepStart {
                        planned_steps: 1,
                        batch_token_count: 16,
                        input_shape: vec![1, 16],
                        target_shape: vec![1, 16],
                    },
                }),
                last_rule_projection_event: None,
            },
            ..TrainingRuntimeArtifact::default()
        };
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
                    experiment: Some(test_experiment_spec()),
                },
                phase_timings: vec![fractal_core::PhaseTiming {
                    phase: fractal_core::RunPhase::Train,
                    elapsed: std::time::Duration::from_secs(1),
                    completed: 1,
                    total: 1,
                }],
                training_runtime,
                execution_outcome: fractal_core::RunExecutionOutcome::InfraFailure,
                quality_outcome: fractal_core::RunQualityOutcome::Clean,
                error: Some("synthetic training failure".to_owned()),
                metrics: None,
            }],
        };

        let report = TournamentRunReport::new(TournamentRunReportParts {
            preset: TournamentPreset::FastTest,
            lane: TournamentLane::Leader,
            comparison: ComparisonContract::authoritative_same_preset(),
            config: TournamentPreset::FastTest.config(),
            species,
            results: Vec::new(),
            artifact,
            bridge_stats: BTreeMap::new(),
        });

        let paths = persist_run_artifacts(&report).unwrap();
        let artifact_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.artifact_path).unwrap()).unwrap();
        let diagnostics = &artifact_json["results"][0]["training_runtime"]["diagnostics"];
        let event_file = diagnostics["structured_output"]["event_file"]
            .as_str()
            .expect("diagnostics event file should be recorded");
        let event_path = PathBuf::from(event_file);
        assert!(event_path.exists());
        let lines = fs::read_to_string(&event_path).unwrap();
        assert!(lines.contains("\"kind\":\"train_step_start\""));
        assert_eq!(
            diagnostics["policy"]["probes"][0]["kind"],
            serde_json::Value::String("train_step".to_owned())
        );
        assert_eq!(
            diagnostics["last_event"]["event"]["kind"],
            serde_json::Value::String("train_step_start".to_owned())
        );

        env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
        env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
        let _ = fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn persisted_artifacts_keep_export_and_failure_snapshot_completeness_separate() {
        let _env_lock = artifact_env_test_lock();
        let temp_root = env::temp_dir().join(format!(
            "fractal-run-artifacts-separation-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&temp_root);
        let artifact_dir = temp_root.join("artifacts");
        let manifest_dir = temp_root.join("manifests");
        env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let mut config = TournamentPreset::FastTest.config();
        config.launch_policy.weight_export = WeightExportPolicy::stage1_default();
        config.launch_policy.failure_snapshot = fractal_core::FailureSnapshotPolicy {
            enabled: true,
            required: true,
            capture_model_weights: false,
            capture_runtime_state: true,
            capture_diagnostics_tail: true,
        };

        let experiment = test_experiment_spec();
        let mut training_runtime = TrainingRuntimeArtifact::empty(&config.launch_policy);
        training_runtime
            .checkpoints
            .push(fractal_core::lifecycle::CheckpointArtifact {
                kind: fractal_core::lifecycle::CheckpointArtifactKind::Latest,
                tokens_seen: 128,
                completed_steps: 2,
                directory: "/tmp/checkpoints/latest".to_owned(),
                long_context_perplexity: Some(1.25),
            });
        training_runtime
            .weight_export
            .record_success(WeightExportArtifact {
                format: WeightExportFormat::BurnBin,
                phase: WeightExportPhase::Best,
                path: "/tmp/exports/best/weights".to_owned(),
                metadata_path: "/tmp/exports/best/metadata.json".to_owned(),
                required: true,
                contract: WeightExportContract {
                    experiment_logical_name: experiment.experiment_id.logical_name.clone(),
                    experiment_run_id: experiment.experiment_id.run_id.clone(),
                    experiment_branch: experiment.experiment_id.branch.clone(),
                    experiment_commit_sha: experiment.experiment_id.commit_sha.clone().unwrap(),
                    species: SpeciesId::P1Contractive.as_str().to_owned(),
                    variant_name: "p1_contractive_v1".to_owned(),
                    model: experiment.model.clone(),
                    vocab_size: config.vocab_size,
                    precision: config.launch_policy.precision.clone(),
                    format: WeightExportFormat::BurnBin,
                },
            });

        let species = species_registry_for_species(SpeciesId::P1Contractive);
        let report = TournamentRunReport::new(TournamentRunReportParts {
            preset: TournamentPreset::FastTest,
            lane: TournamentLane::Leader,
            comparison: ComparisonContract::authoritative_same_preset(),
            config: config.clone(),
            species,
            results: vec![RankedSpeciesResult {
                rank: 1,
                species: SpeciesId::P1Contractive,
                stability_score: 0.53,
                long_context_perplexity: 1.54,
                arc_accuracy: 0.68,
                tokens_per_sec: 100.0,
                fitness: 0.58,
            }],
            artifact: fractal_core::TournamentRunArtifact {
                config: config.clone(),
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
                        config,
                        experiment: Some(experiment),
                    },
                    phase_timings: vec![fractal_core::PhaseTiming {
                        phase: fractal_core::RunPhase::Train,
                        elapsed: std::time::Duration::from_secs(1),
                        completed: 1,
                        total: 1,
                    }],
                    training_runtime,
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
            },
            bridge_stats: BTreeMap::new(),
        });

        let paths = persist_run_artifacts(&report).unwrap();
        let artifact_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.artifact_path).unwrap()).unwrap();
        let runtime = &artifact_json["results"][0]["training_runtime"];

        assert_eq!(
            runtime["weight_export"]["missing_required_phases"],
            serde_json::json!(["final"])
        );
        assert_eq!(
            runtime["weight_export"]["completeness"],
            serde_json::Value::String("partial".to_owned())
        );
        assert_eq!(
            runtime["failure_snapshot"]["attempted"],
            serde_json::Value::Bool(false)
        );
        assert_eq!(
            runtime["failure_snapshot"]["completeness"],
            serde_json::Value::String("not-captured".to_owned())
        );

        env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
        env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
        let _ = fs::remove_dir_all(&temp_root);
    }

    fn test_experiment_spec() -> ExperimentSpec {
        let config = TournamentPreset::FastTest.config();
        ExperimentSpec {
            experiment_id: ExperimentId {
                logical_name: "fast-test-run".to_owned(),
                run_id: "run-123".to_owned(),
                branch: Some("codex/exp-spec-core".to_owned()),
                commit_sha: Some("abc123".to_owned()),
                created_at_unix_ms: 123,
            },
            question: ExperimentQuestion {
                summary: "evaluate p1 contractive on fast-test".to_owned(),
                lane_intent: LaneIntent::Benchmark,
                decision_intent: DecisionIntent::Benchmark,
            },
            variant: VariantSpec {
                species: SpeciesId::P1Contractive,
                variant_name: fractal_core::PrimitiveVariantName::new_unchecked(
                    "p1_contractive_v1",
                ),
            },
            model: fractal_core::ModelContractSpec::recursive_kernel_v1(
                config.dim,
                config.max_recursion_depth,
            ),
            training_input: TrainingInputSpec::synthetic(),
            budget: BudgetSpec::from_config(TournamentPreset::FastTest, &config),
            optimizer: OptimizerSpec::legacy_adam(config.learning_rate),
            runtime: RuntimeSurfaceSpec::default(),
            comparison: ComparisonContract::authoritative_same_preset(),
            execution: ExecutionTarget {
                kind: ExecutionTargetKind::Local,
                backend: ExecutionBackend::from_compute_backend(&config.execution_backend),
                execution_mode: config.execution_mode,
                pod_id: None,
                wrapper_timeout_seconds: None,
            },
            artifacts: ArtifactPolicy::default(),
        }
    }
}
