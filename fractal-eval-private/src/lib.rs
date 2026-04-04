pub mod v2_ablation;
pub mod v2_benchmark;
pub mod v2_checkpoint;
pub mod v2_fixture;
pub mod v2_ledger;
pub mod v2_synthetic;
pub mod v2_training;

pub use v2_ablation::{
    required_v2_ablation_modes, run_required_v2_ablation_sweep, V2AblationCaseReport,
    V2AblationConfig, V2AblationReport, V2RootTopology,
};
pub use v2_benchmark::{
    run_baseline_v2_benchmark_suite, run_v2_benchmark_suite_for_model, V2BenchmarkConfig,
    V2BenchmarkEntry, V2BenchmarkReport, V2BenchmarkSurface, V2LeafUsageBin,
    V2ObservabilitySnapshot, DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS,
};
pub use v2_checkpoint::{
    load_baseline_v2_checkpoint_model, load_v2_smoke_train_report, LoadedV2CheckpointModel,
    V2CheckpointSelection,
};
pub use v2_fixture::{
    build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig,
};
pub use v2_ledger::{
    append_v2_results_ledger_entry, default_v2_results_ledger_path, read_v2_results_ledger,
    resolve_requested_v2_results_ledger_path, V2ResultsLedgerEntry, V2ResultsLedgerKind,
    DEFAULT_V2_RESULTS_LEDGER_PATH,
};
pub use v2_synthetic::{
    default_v2_synthetic_probe_suites, run_v2_synthetic_probe_suite,
    run_v2_synthetic_probe_suite_with_modes, run_v2_synthetic_probe_suites,
    run_v2_synthetic_probe_suites_with_modes, SyntheticProbeKind, SyntheticProbeMetrics,
    SyntheticProbeMode, SyntheticProbeModeReport, SyntheticProbeModel, SyntheticProbeReport,
    SyntheticProbeSample, SyntheticProbeSampleResult, SyntheticProbeSuite,
    SyntheticProbeSuiteReport,
};
pub use v2_training::{
    baseline_v2_byte_level_smoke_model_config, default_v2_smoke_corpus_paths,
    run_baseline_v2_smoke_train, run_v2_smoke_train_with_model, ByteLevelVocabularyContract,
    V2SmokeCheckpointArtifacts, V2SmokeCheckpointKind, V2SmokeCorpusStats, V2SmokeEvalMetrics,
    V2SmokeTrainConfig, V2SmokeTrainModel, V2SmokeTrainReport, V2SmokeTrainResult,
    V2SmokeTrainStepReport, BYTE_LEVEL_PAD_TOKEN, BYTE_LEVEL_VOCAB_SIZE,
    DEFAULT_V2_SMOKE_BATCH_SIZE, DEFAULT_V2_SMOKE_EVAL_BATCHES,
    DEFAULT_V2_SMOKE_EVAL_HOLDOUT_EVERY, DEFAULT_V2_SMOKE_LEARNING_RATE, DEFAULT_V2_SMOKE_SEQ_LEN,
    DEFAULT_V2_SMOKE_TRAIN_STEPS, DEFAULT_V2_SMOKE_WINDOW_STRIDE,
};

use fractal_core::{RankedSpeciesResult, SpeciesRawMetrics};

pub fn stability_score(grad_norm_depth_20: f64) -> f64 {
    if !grad_norm_depth_20.is_finite() {
        0.0
    } else {
        1.0 / (1.0 + grad_norm_depth_20)
    }
}

pub fn perplexity_score(perplexity: f64) -> f64 {
    if !perplexity.is_finite() || perplexity <= 0.0 {
        0.0
    } else {
        1.0 / perplexity.max(1.0)
    }
}

pub fn speed_score(tokens_per_sec: f64, best_tokens_per_sec: f64) -> f64 {
    if !tokens_per_sec.is_finite()
        || tokens_per_sec <= 0.0
        || !best_tokens_per_sec.is_finite()
        || best_tokens_per_sec <= f64::EPSILON
    {
        0.0
    } else {
        (tokens_per_sec / best_tokens_per_sec).clamp(0.0, 1.0)
    }
}

pub fn aggregate_results(mut metrics: Vec<SpeciesRawMetrics>) -> Vec<RankedSpeciesResult> {
    let best_tokens = metrics
        .iter()
        .map(|metric| metric.tokens_per_sec)
        .filter(|tokens_per_sec| tokens_per_sec.is_finite() && *tokens_per_sec > 0.0)
        .fold(0.0f64, f64::max);

    let mut ranked = metrics
        .drain(..)
        .map(|metric| {
            let stability = stability_score(metric.grad_norm_depth_20);
            let perplexity_component = perplexity_score(metric.long_context_perplexity);
            let accuracy = accuracy_score(metric.arc_accuracy);
            let speed = speed_score(metric.tokens_per_sec, best_tokens);
            let fitness =
                0.35 * stability + 0.30 * perplexity_component + 0.25 * accuracy + 0.10 * speed;

            RankedSpeciesResult {
                rank: 0,
                species: metric.species,
                stability_score: stability,
                long_context_perplexity: metric.long_context_perplexity,
                arc_accuracy: metric.arc_accuracy,
                tokens_per_sec: metric.tokens_per_sec,
                fitness,
            }
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .fitness
            .partial_cmp(&left.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (index, result) in ranked.iter_mut().enumerate() {
        result.rank = index + 1;
    }

    ranked
}

fn accuracy_score(accuracy: f64) -> f64 {
    if !accuracy.is_finite() {
        0.0
    } else {
        accuracy.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::aggregate_results;
    use fractal_core::{SpeciesId, SpeciesRawMetrics};

    #[test]
    fn aggregate_results_penalizes_non_finite_perplexity() {
        let ranked = aggregate_results(vec![
            SpeciesRawMetrics {
                species: SpeciesId::P1Contractive,
                grad_norm_depth_20: 1.0,
                long_context_perplexity: f64::NAN,
                arc_accuracy: 0.0,
                tokens_per_sec: 100.0,
            },
            SpeciesRawMetrics {
                species: SpeciesId::Ifs,
                grad_norm_depth_20: 1.0,
                long_context_perplexity: 20.0,
                arc_accuracy: 0.0,
                tokens_per_sec: 100.0,
            },
        ]);

        assert_eq!(ranked[0].species, SpeciesId::Ifs);
        assert_eq!(ranked[1].species, SpeciesId::P1Contractive);
    }
}
