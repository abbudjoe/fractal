use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use fractal_core::error::FractalError;

use crate::{SyntheticProbeReport, V2AblationReport, V2BenchmarkReport, V2SmokeTrainReport};

pub const DEFAULT_V2_RESULTS_LEDGER_PATH: &str = "docs/v2-results-ledger.jsonl";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum V2ResultsLedgerKind {
    SmokeTrain,
    SyntheticProbe,
    Benchmark,
    AblationSweep,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V2ResultsLedgerEntry {
    pub schema_version: u32,
    pub recorded_at_unix_seconds: u64,
    pub kind: V2ResultsLedgerKind,
    pub model: String,
    pub note: String,
    pub run_label: Option<String>,
    pub payload: Value,
}

impl V2ResultsLedgerEntry {
    pub fn smoke_train(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &V2SmokeTrainReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Self::from_payload(
            V2ResultsLedgerKind::SmokeTrain,
            model,
            note,
            run_label,
            report,
        )
    }

    pub fn synthetic_probe(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &SyntheticProbeReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Self::from_payload(
            V2ResultsLedgerKind::SyntheticProbe,
            model,
            note,
            run_label,
            report,
        )
    }

    pub fn benchmark(
        report: &V2BenchmarkReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Self::from_payload(
            V2ResultsLedgerKind::Benchmark,
            report.model.clone(),
            report.note.clone(),
            run_label,
            report,
        )
    }

    pub fn ablation_sweep(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &V2AblationReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Self::from_payload(
            V2ResultsLedgerKind::AblationSweep,
            model,
            note,
            run_label,
            report,
        )
    }

    fn from_payload(
        kind: V2ResultsLedgerKind,
        model: impl Into<String>,
        note: impl Into<String>,
        run_label: Option<String>,
        payload: impl Serialize,
    ) -> Result<Self, FractalError> {
        Ok(Self {
            schema_version: 1,
            recorded_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            kind,
            model: model.into(),
            note: note.into(),
            run_label,
            payload: serde_json::to_value(payload).map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to serialize v2 results ledger payload: {error}"
                ))
            })?,
        })
    }
}

pub fn default_v2_results_ledger_path(repo_root: impl AsRef<Path>) -> PathBuf {
    repo_root.as_ref().join(DEFAULT_V2_RESULTS_LEDGER_PATH)
}

pub fn resolve_requested_v2_results_ledger_path(
    repo_root: impl AsRef<Path>,
    request: Option<&str>,
) -> Result<Option<PathBuf>, FractalError> {
    let Some(request) = request else {
        return Ok(None);
    };
    if request.trim().is_empty() {
        return Err(FractalError::InvalidConfig(
            "v2_results_ledger.path must not be empty".to_string(),
        ));
    }
    if request == "default" {
        return Ok(Some(default_v2_results_ledger_path(repo_root)));
    }

    Ok(Some(PathBuf::from(request)))
}

pub fn append_v2_results_ledger_entry(
    path: impl AsRef<Path>,
    entry: &V2ResultsLedgerEntry,
) -> Result<(), FractalError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to create v2 results ledger directory {}: {error}",
                parent.display()
            ))
        })?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to open v2 results ledger {} for append: {error}",
                path.display()
            ))
        })?;
    serde_json::to_writer(&mut file, entry).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize v2 results ledger entry for {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to terminate v2 results ledger entry for {}: {error}",
            path.display()
        ))
    })
}

pub fn read_v2_results_ledger(
    path: impl AsRef<Path>,
) -> Result<Vec<V2ResultsLedgerEntry>, FractalError> {
    let path = path.as_ref();
    let file = fs::File::open(path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to open v2 results ledger {}: {error}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for (index, line) in reader.lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read line {line_number} from v2 results ledger {}: {error}",
                path.display()
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<V2ResultsLedgerEntry>(&line).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to parse line {line_number} from v2 results ledger {}: {error}",
                path.display()
            ))
        })?;
        entries.push(entry);
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        baseline_v2_byte_level_smoke_model_config, SyntheticProbeKind, SyntheticProbeMetrics,
        SyntheticProbeMode, SyntheticProbeModeReport, SyntheticProbeReport,
        SyntheticProbeSuiteReport, V2AblationCaseReport, V2BenchmarkConfig, V2BenchmarkEntry,
        V2BenchmarkReport, V2BenchmarkSurface, V2LeafUsageBin, V2ObservabilitySnapshot,
        V2RootTopology, V2SmokeCheckpointArtifacts, V2SmokeCheckpointKind, V2SmokeCorpusStats,
        V2SmokeEvalMetrics, V2SmokeTrainConfig, V2SmokeTrainReport, V2SmokeTrainStepReport,
    };

    #[test]
    fn append_and_read_round_trip_preserves_entry_order() {
        let root = unique_temp_dir("v2-results-ledger-roundtrip");
        let ledger_path = root.join("ledger.jsonl");
        let smoke = V2ResultsLedgerEntry::smoke_train(
            "baseline_v2_byte_level_smoke_cpu_candle",
            "small byte-level v2 smoke training run on real local text",
            &sample_smoke_report(),
            Some("smoke-run".to_string()),
        )
        .unwrap();
        let benchmark =
            V2ResultsLedgerEntry::benchmark(&sample_benchmark_report(), Some("bench".to_string()))
                .unwrap();

        append_v2_results_ledger_entry(&ledger_path, &smoke).unwrap();
        append_v2_results_ledger_entry(&ledger_path, &benchmark).unwrap();

        let entries = read_v2_results_ledger(&ledger_path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, V2ResultsLedgerKind::SmokeTrain);
        assert_eq!(entries[1].kind, V2ResultsLedgerKind::Benchmark);
        assert_eq!(entries[0].run_label.as_deref(), Some("smoke-run"));
        assert_eq!(entries[1].run_label.as_deref(), Some("bench"));
    }

    #[test]
    fn default_path_resolves_under_repo_docs() {
        let path = default_v2_results_ledger_path("/tmp/fractal");
        assert_eq!(
            path,
            PathBuf::from("/tmp/fractal/docs/v2-results-ledger.jsonl")
        );
    }

    #[test]
    fn requested_default_path_resolves_under_repo_docs() {
        let path =
            resolve_requested_v2_results_ledger_path("/tmp/fractal", Some("default")).unwrap();
        assert_eq!(
            path,
            Some(PathBuf::from("/tmp/fractal/docs/v2-results-ledger.jsonl"))
        );
    }

    #[test]
    fn requested_custom_path_round_trips_verbatim() {
        let path = resolve_requested_v2_results_ledger_path(
            "/tmp/fractal",
            Some("/tmp/custom-ledger.jsonl"),
        )
        .unwrap();
        assert_eq!(path, Some(PathBuf::from("/tmp/custom-ledger.jsonl")));
    }

    #[test]
    fn smoke_train_entry_payload_contains_eval_summary() {
        let entry = V2ResultsLedgerEntry::smoke_train(
            "baseline_v2_byte_level_smoke_cpu_candle",
            "small byte-level v2 smoke training run on real local text",
            &sample_smoke_report(),
            None,
        )
        .unwrap();

        assert_eq!(entry.kind, V2ResultsLedgerKind::SmokeTrain);
        assert_eq!(
            entry.payload["final_eval"]["mean_loss"],
            serde_json::json!(5.49)
        );
    }

    #[test]
    fn ablation_entry_payload_contains_cases() {
        let entry = V2ResultsLedgerEntry::ablation_sweep(
            "baseline_v2_required_ablation_cpu_candle",
            "equal-budget single-root and multi-root ablation sweep",
            &sample_ablation_report(),
            Some("ablation".to_string()),
        )
        .unwrap();

        assert_eq!(entry.kind, V2ResultsLedgerKind::AblationSweep);
        assert_eq!(entry.payload["cases"].as_array().unwrap().len(), 1);
        assert_eq!(
            entry.payload["cases"][0]["topology"],
            serde_json::json!("single_root")
        );
    }

    fn sample_smoke_report() -> V2SmokeTrainReport {
        V2SmokeTrainReport {
            config: V2SmokeTrainConfig::new(
                vec![PathBuf::from("/tmp/corpus.md")],
                PathBuf::from("/tmp/artifacts"),
            ),
            corpus: V2SmokeCorpusStats {
                files: vec![PathBuf::from("/tmp/corpus.md")],
                total_bytes: 128,
                total_sequences: 8,
                train_sequences: 6,
                eval_sequences: 2,
                seq_len: 32,
                window_stride: 32,
            },
            initial_eval: V2SmokeEvalMetrics {
                batch_count: 1,
                mean_loss: 5.6,
                perplexity: 270.0,
            },
            final_eval: V2SmokeEvalMetrics {
                batch_count: 1,
                mean_loss: 5.49,
                perplexity: 242.0,
            },
            best_eval: V2SmokeEvalMetrics {
                batch_count: 1,
                mean_loss: 5.49,
                perplexity: 242.0,
            },
            best_checkpoint_kind: V2SmokeCheckpointKind::FinalEval,
            train_steps: vec![V2SmokeTrainStepReport {
                step: 1,
                learning_rate: 1e-3,
                train_loss: 5.5,
                train_perplexity: 245.0,
                seen_tokens: 32,
            }],
            checkpoint: V2SmokeCheckpointArtifacts {
                directory: PathBuf::from("/tmp/artifacts"),
                final_model_path: PathBuf::from("/tmp/artifacts/final-model.bin"),
                best_model_path: PathBuf::from("/tmp/artifacts/best-model.bin"),
                final_optimizer_path: PathBuf::from("/tmp/artifacts/final-optimizer.bin"),
                best_optimizer_path: PathBuf::from("/tmp/artifacts/best-optimizer.bin"),
                report_path: PathBuf::from("/tmp/artifacts/report.json"),
            },
        }
    }

    fn sample_synthetic_report() -> SyntheticProbeReport {
        SyntheticProbeReport {
            suites: vec![SyntheticProbeSuiteReport {
                kind: SyntheticProbeKind::Copy,
                sample_count: 1,
                mode_reports: vec![SyntheticProbeModeReport {
                    mode: SyntheticProbeMode::TreePlusExactRead,
                    metrics: SyntheticProbeMetrics {
                        accuracy: 1.0,
                        mean_target_logit: 2.0,
                        mean_loss: 0.25,
                    },
                    sample_results: Vec::new(),
                }],
            }],
        }
    }

    fn sample_benchmark_report() -> V2BenchmarkReport {
        V2BenchmarkReport {
            model: "baseline_v2_random_init".to_string(),
            note: "benchmark note".to_string(),
            config: V2BenchmarkConfig {
                sequence_lengths: vec![256],
                leaf_size: 16,
                iterations: 1,
                warmup_iterations: 0,
            },
            entries: vec![V2BenchmarkEntry {
                surface: V2BenchmarkSurface::ForwardPass,
                sequence_length: 256,
                iterations: 1,
                warmup_iterations: 0,
                logical_tokens_per_iteration: 256,
                total_wall_time_ms: 4.0,
                mean_wall_time_ms: 4.0,
                tokens_per_sec: 64.0,
                peak_rss_bytes: 1_000,
                peak_rss_delta_bytes: 100,
                observability: V2ObservabilitySnapshot {
                    routing_sparsity: 0.5,
                    root_collapse_mean_pairwise_cosine_similarity: 0.8,
                    exact_read_usage: 0.25,
                    mean_retrieval_distance: 32.0,
                    tree_depth_reached: 2,
                    level0_leaf_count: 16,
                    head_agreement_rate: 0.5,
                    has_dead_or_unused_tree_nodes: false,
                    selected_leaf_usage: vec![V2LeafUsageBin {
                        leaf_index: 0,
                        count: 1,
                    }],
                },
            }],
        }
    }

    fn sample_ablation_report() -> V2AblationReport {
        V2AblationReport {
            note: "ablation note".to_string(),
            cases: vec![V2AblationCaseReport {
                topology: V2RootTopology::SingleRoot,
                model_config: baseline_v2_byte_level_smoke_model_config(),
                synthetic: sample_synthetic_report(),
            }],
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
