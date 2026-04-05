use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use fractal_core::{
    error::FractalError, phase1_hybrid_attention_baseline_matrix, HybridAttentionBaselineMatrix,
    HybridAttentionComparisonContract, HybridAttentionEfficiencyTarget,
};

pub const DEFAULT_V3A_RESULTS_LEDGER_PATH: &str = "docs/v3a-results-ledger.jsonl";
pub const DEFAULT_V3A_SMOKE_TRAIN_STEPS: usize = 128;
pub const DEFAULT_V3A_SMOKE_EVAL_BATCHES: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridAttentionMatrixConfig {
    pub corpus_paths: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub baseline_matrix: HybridAttentionBaselineMatrix,
    pub train_steps: usize,
    pub eval_batches: usize,
}

impl HybridAttentionMatrixConfig {
    pub fn new(corpus_paths: Vec<PathBuf>, output_dir: PathBuf) -> Self {
        Self {
            corpus_paths,
            output_dir,
            baseline_matrix: phase1_hybrid_attention_baseline_matrix(),
            train_steps: DEFAULT_V3A_SMOKE_TRAIN_STEPS,
            eval_batches: DEFAULT_V3A_SMOKE_EVAL_BATCHES,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.corpus_paths.is_empty() {
            return Err(FractalError::InvalidConfig(
                "hybrid_attention_matrix.corpus_paths must include at least one file".to_string(),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::{HybridAttentionMatrixConfig, HybridAttentionMatrixPlan};

    #[test]
    fn phase1_plan_points_to_dedicated_runner_surface() {
        let plan = HybridAttentionMatrixPlan::phase1_default();
        assert_eq!(plan.runner.binary, "src/bin/v3a-hybrid-attention-matrix.rs");
    }

    #[test]
    fn matrix_config_requires_real_corpus_paths() {
        let config = HybridAttentionMatrixConfig::new(vec![], "tmp".into());
        assert!(config.validate().is_err());
    }
}
