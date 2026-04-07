use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use fractal_core::{
    error::FractalError, phase1_hybrid_attention_baseline_matrix, HybridAttentionBaselineMatrix,
    HybridAttentionComparisonContract, HybridAttentionEfficiencyTarget,
};

use crate::{
    hybrid_attention_training::HybridAttentionMatrixVariantOutcome, ByteLevelSmokeCorpusSource,
};

pub const DEFAULT_V3A_RESULTS_LEDGER_PATH: &str = "docs/v3a-results-ledger.jsonl";
pub const DEFAULT_V3A_SMOKE_TRAIN_STEPS: usize = 128;
pub const DEFAULT_V3A_SMOKE_EVAL_BATCHES: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionMatrixConfig {
    pub corpus_source: ByteLevelSmokeCorpusSource,
    pub output_dir: PathBuf,
    pub baseline_matrix: HybridAttentionBaselineMatrix,
    pub train_steps: usize,
    pub eval_batches: usize,
}

impl HybridAttentionMatrixConfig {
    pub fn new(corpus_source: ByteLevelSmokeCorpusSource, output_dir: PathBuf) -> Self {
        Self {
            corpus_source,
            output_dir,
            baseline_matrix: phase1_hybrid_attention_baseline_matrix(),
            train_steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.train_steps == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_matrix.train_steps must be greater than zero".to_string(),
            ));
        }
        if self.eval_batches == 0 {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_matrix.eval_batches must be greater than zero".to_string(),
            ));
        }
        self.corpus_source.validate()?;
        self.baseline_matrix.validate()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionRunnerLocation {
    pub core_module: String,
    pub eval_module: String,
    pub binary: String,
}

impl HybridAttentionRunnerLocation {
    pub fn phase1_default() -> Self {
        Self {
            core_module: "fractal-core/src/hybrid_attention/".to_string(),
            eval_module: "fractal-eval-private/src/hybrid_attention.rs".to_string(),
            binary: "src/bin/v3a-hybrid-attention-matrix.rs".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionMatrixPlan {
    pub question: String,
    pub comparison: HybridAttentionComparisonContract,
    pub primary_efficiency_target: HybridAttentionEfficiencyTarget,
    pub runner: HybridAttentionRunnerLocation,
}

impl HybridAttentionMatrixPlan {
    pub fn phase1_default() -> Self {
        let matrix = phase1_hybrid_attention_baseline_matrix();
        let comparison = matrix.comparison.clone();
        Self {
            question: "Can we validate a faithful Rust Mamba-3-style hybrid baseline before treating our primitive line as a contender inside the same attention-centric predictive backbone?".to_string(),
            primary_efficiency_target: comparison.primary_efficiency_target,
            comparison,
            runner: HybridAttentionRunnerLocation::phase1_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridAttentionMatrixLedgerReport {
    pub requested_variant: String,
    pub note: String,
    pub variants: Vec<HybridAttentionMatrixVariantOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum V3aResultsLedgerKind {
    Path1MatrixRun,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct V3aResultsLedgerEntry {
    pub schema_version: u32,
    pub recorded_at_unix_seconds: u64,
    pub kind: V3aResultsLedgerKind,
    pub model: String,
    pub note: String,
    pub run_label: Option<String>,
    pub payload: Value,
}

impl V3aResultsLedgerEntry {
    pub fn path1_matrix_run(
        model: impl Into<String>,
        note: impl Into<String>,
        report: &HybridAttentionMatrixLedgerReport,
        run_label: Option<String>,
    ) -> Result<Self, FractalError> {
        Ok(Self {
            schema_version: 1,
            recorded_at_unix_seconds: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            kind: V3aResultsLedgerKind::Path1MatrixRun,
            model: model.into(),
            note: note.into(),
            run_label,
            payload: serde_json::to_value(report).map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to serialize v3a results ledger payload: {error}"
                ))
            })?,
        })
    }
}

pub fn default_v3a_results_ledger_path(repo_root: impl AsRef<Path>) -> PathBuf {
    repo_root.as_ref().join(DEFAULT_V3A_RESULTS_LEDGER_PATH)
}

pub fn resolve_requested_v3a_results_ledger_path(
    repo_root: impl AsRef<Path>,
    request: Option<&str>,
) -> Result<Option<PathBuf>, FractalError> {
    let Some(request) = request else {
        return Ok(None);
    };
    if request.trim().is_empty() {
        return Err(FractalError::InvalidConfig(
            "v3a_results_ledger.path must not be empty".to_string(),
        ));
    }
    if request == "default" {
        return Ok(Some(default_v3a_results_ledger_path(repo_root)));
    }
    Ok(Some(PathBuf::from(request)))
}

pub fn append_v3a_results_ledger_entry(
    path: impl AsRef<Path>,
    entry: &V3aResultsLedgerEntry,
) -> Result<(), FractalError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to create v3a results ledger directory {}: {error}",
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
                "failed to open v3a results ledger {} for append: {error}",
                path.display()
            ))
        })?;
    serde_json::to_writer(&mut file, entry).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to serialize v3a results ledger entry for {}: {error}",
            path.display()
        ))
    })?;
    file.write_all(b"\n").map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to terminate v3a results ledger entry for {}: {error}",
            path.display()
        ))
    })
}

pub fn read_v3a_results_ledger(
    path: impl AsRef<Path>,
) -> Result<Vec<V3aResultsLedgerEntry>, FractalError> {
    let path = path.as_ref();
    let file = fs::File::open(path).map_err(|error| {
        FractalError::InvalidState(format!(
            "failed to open v3a results ledger {}: {error}",
            path.display()
        ))
    })?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();
    for (line_index, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read v3a results ledger line {} from {}: {error}",
                line_index + 1,
                path.display()
            ))
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let entry = serde_json::from_str::<V3aResultsLedgerEntry>(&line).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to parse v3a results ledger line {} from {}: {error}",
                line_index + 1,
                path.display()
            ))
        })?;
        entries.push(entry);
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::{
        append_v3a_results_ledger_entry, default_v3a_results_ledger_path, read_v3a_results_ledger,
        resolve_requested_v3a_results_ledger_path, HybridAttentionMatrixConfig,
        HybridAttentionMatrixLedgerReport, HybridAttentionMatrixPlan, V3aResultsLedgerEntry,
        V3aResultsLedgerKind,
    };
    use crate::{ByteLevelSmokeCorpusSource, HybridAttentionMatrixVariantOutcome};

    #[test]
    fn phase1_plan_points_to_dedicated_runner_surface() {
        let plan = HybridAttentionMatrixPlan::phase1_default();
        assert_eq!(plan.runner.binary, "src/bin/v3a-hybrid-attention-matrix.rs");
    }

    #[test]
    fn matrix_config_requires_real_corpus_paths() {
        let config = HybridAttentionMatrixConfig::new(
            ByteLevelSmokeCorpusSource::raw_files(vec![]),
            "tmp".into(),
        );
        assert!(config.validate().is_err());
    }

    #[test]
    fn default_v3a_ledger_path_points_to_docs_jsonl() {
        let path = default_v3a_results_ledger_path("/tmp/fractal");
        assert_eq!(
            path,
            PathBuf::from("/tmp/fractal/docs/v3a-results-ledger.jsonl")
        );
    }

    #[test]
    fn resolve_requested_v3a_ledger_supports_default() {
        let path =
            resolve_requested_v3a_results_ledger_path("/tmp/fractal", Some("default")).unwrap();
        assert_eq!(
            path,
            Some(PathBuf::from("/tmp/fractal/docs/v3a-results-ledger.jsonl"))
        );
    }

    #[test]
    fn append_and_read_v3a_ledger_round_trip() {
        let root = std::env::temp_dir().join(format!("fractal-v3a-ledger-{}", std::process::id()));
        let ledger_path = root.join("ledger.jsonl");
        let report = HybridAttentionMatrixLedgerReport {
            requested_variant: "all".to_string(),
            note: "matrix".to_string(),
            variants: vec![HybridAttentionMatrixVariantOutcome::Skipped {
                label: "attention-only".to_string(),
                kind: fractal_core::HybridAttentionVariantKind::AttentionOnly,
                reason: "not requested".to_string(),
            }],
        };
        let entry = V3aResultsLedgerEntry::path1_matrix_run(
            "v3a_hybrid_attention_matrix",
            "note",
            &report,
            Some("smoke".to_string()),
        )
        .unwrap();
        append_v3a_results_ledger_entry(&ledger_path, &entry).unwrap();
        let entries = read_v3a_results_ledger(&ledger_path).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, V3aResultsLedgerKind::Path1MatrixRun);
        assert_eq!(entries[0].run_label.as_deref(), Some("smoke"));
        let _ = fs::remove_file(ledger_path);
        let _ = fs::remove_dir_all(root);
    }
}
