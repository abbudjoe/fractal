pub mod hybrid_attention;
pub mod hybrid_attention_training;
pub mod hybrid_prevalidation;
pub mod hybrid_training;
pub mod v2_ablation;
pub mod v2_benchmark;
pub mod v2_checkpoint;
pub mod v2_fixture;
pub mod v2_learned_ablation;
pub mod v2_ledger;
pub mod v2_supervised;
pub mod v2_synthetic;
pub mod v2_training;

pub use hybrid_attention::{
    append_v3a_results_ledger_entry, default_v3a_results_ledger_path, read_v3a_results_ledger,
    resolve_requested_v3a_results_ledger_path, HybridAttentionMatrixConfig,
    HybridAttentionMatrixLedgerReport, HybridAttentionMatrixPlan, HybridAttentionRunnerLocation,
    V3aResultsLedgerEntry, V3aResultsLedgerKind, DEFAULT_V3A_RESULTS_LEDGER_PATH,
};
pub use hybrid_attention_training::{
    run_attention_only_hybrid_attention_smoke_train, run_primitive_hybrid_attention_smoke_train,
    run_reference_ssm_hybrid_attention_smoke_train, HybridAttentionExecutionBackend,
    HybridAttentionMatrixVariantOutcome, HybridAttentionRuntimeMetrics,
    HybridAttentionSmokeTrainConfig, HybridAttentionSmokeTrainReport, DEFAULT_V3A_SMOKE_BATCH_SIZE,
    DEFAULT_V3A_SMOKE_EVAL_BATCHES, DEFAULT_V3A_SMOKE_EVAL_HOLDOUT_EVERY,
    DEFAULT_V3A_SMOKE_LEARNING_RATE, DEFAULT_V3A_SMOKE_SEED, DEFAULT_V3A_SMOKE_SEQ_LEN,
    DEFAULT_V3A_SMOKE_TRAIN_STEPS, DEFAULT_V3A_SMOKE_WINDOW_STRIDE,
    HYBRID_ATTENTION_RUNTIME_MEMORY_NOTE,
};
pub use hybrid_prevalidation::{
    append_hybrid_results_ledger_entry, build_baseline_hybrid_rescue_model,
    default_hybrid_rescue_prevalidation_suites, default_hybrid_results_ledger_path,
    hybrid_rescue_prevalidation_suites_for_leaf_size, resolve_requested_hybrid_results_ledger_path,
    run_baseline_hybrid_rescue_prevalidation, run_hybrid_rescue_prevalidation_suite_with_modes,
    run_hybrid_rescue_prevalidation_with_modes, BaselineHybridRescueModel,
    BaselineHybridRescueModelConfig, HybridRescueMetrics, HybridRescueModeReport,
    HybridRescuePrevalidationReport, HybridRescueProbeMode, HybridRescueProbeSuite,
    HybridRescueSampleResult, HybridRescueSuiteKind, HybridRescueSuiteReport,
    HybridResultsLedgerEntry, HybridResultsLedgerKind,
};
pub use hybrid_training::{
    hybrid_eval_metrics_for_mode, run_baseline_hybrid_rescue_frozen_train,
    run_hybrid_rescue_frozen_train_with_model, HybridRescueFrozenEvalMetrics,
    HybridRescueFrozenEvalModeSet, HybridRescueFrozenSplitStats, HybridRescueFrozenSuiteSplit,
    HybridRescueFrozenTrainConfig, HybridRescueFrozenTrainReport, HybridRescueFrozenTrainResult,
    HybridRescueFrozenTrainStepReport, DEFAULT_HYBRID_RESCUE_FROZEN_EVAL_HOLDOUT_EVERY,
    DEFAULT_HYBRID_RESCUE_FROZEN_LEARNING_RATE, DEFAULT_HYBRID_RESCUE_FROZEN_STEPS,
};
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
pub use v2_learned_ablation::{
    required_v2_learned_ablation_modes, run_required_v2_learned_ablation_matrix,
    V2LearnedAblationConfig, V2LearnedAblationReport, V2LearnedAblationStepReport,
    V2LearnedAblationSuiteMetrics, V2LearnedAblationTopologyReport, V2RequiredAblationStep,
};
pub use v2_ledger::{
    append_v2_results_ledger_entry, default_v2_results_ledger_path, read_v2_results_ledger,
    resolve_requested_v2_results_ledger_path, V2ResultsLedgerEntry, V2ResultsLedgerKind,
    DEFAULT_V2_RESULTS_LEDGER_PATH,
};
pub use v2_supervised::{
    filter_synthetic_probe_suites, mode_eval_summary_by_kind,
    run_baseline_v2_supervised_synthetic_train,
    run_baseline_v2_supervised_synthetic_train_with_modes,
    run_v2_supervised_synthetic_train_with_model,
    run_v2_supervised_synthetic_train_with_model_and_modes, V2SupervisedSyntheticEvalMetrics,
    V2SupervisedSyntheticSplitStats, V2SupervisedSyntheticSuiteSplit,
    V2SupervisedSyntheticTrainConfig, V2SupervisedSyntheticTrainModel,
    V2SupervisedSyntheticTrainReport, V2SupervisedSyntheticTrainResult,
    V2SupervisedSyntheticTrainStepReport, DEFAULT_V2_SUPERVISED_SYNTHETIC_EVAL_HOLDOUT_EVERY,
    DEFAULT_V2_SUPERVISED_SYNTHETIC_LEARNING_RATE, DEFAULT_V2_SUPERVISED_SYNTHETIC_STEPS,
    V2_SUPERVISED_SYNTHETIC_LEAF_SIZE,
};
pub use v2_synthetic::{
    default_v2_synthetic_probe_suites, run_v2_synthetic_probe_suite,
    run_v2_synthetic_probe_suite_with_modes, run_v2_synthetic_probe_suites,
    run_v2_synthetic_probe_suites_with_modes,
    run_v2_synthetic_projection_diagnostic_suite_with_modes,
    run_v2_synthetic_projection_diagnostic_suites_with_modes,
    v2_synthetic_probe_suites_for_leaf_size, SyntheticProbeKind, SyntheticProbeMetrics,
    SyntheticProbeMode, SyntheticProbeModeReport, SyntheticProbeModel,
    SyntheticProbeProjectionMetrics, SyntheticProbeProjectionModeReport,
    SyntheticProbeProjectionReport, SyntheticProbeProjectionSampleResult,
    SyntheticProbeProjectionSuiteReport, SyntheticProbeReport, SyntheticProbeSample,
    SyntheticProbeSampleResult, SyntheticProbeSuite, SyntheticProbeSuiteReport,
};
pub use v2_training::{
    baseline_v2_byte_level_smoke_model_config, byte_level_smoke_corpus_stats_from_source,
    default_v2_smoke_corpus_paths, default_v3a_fineweb_stage0_canary_corpus_source,
    run_baseline_v2_smoke_train, run_v2_smoke_train_with_model, ByteLevelSmokeCorpusSource,
    ByteLevelVocabularyContract, V2CheckpointArtifacts, V2CheckpointKind,
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
