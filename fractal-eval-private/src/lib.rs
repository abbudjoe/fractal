pub mod v2_benchmark;
pub mod v2_fixture;
pub mod v2_synthetic;

pub use v2_benchmark::{
    run_baseline_v2_benchmark_suite, V2BenchmarkConfig, V2BenchmarkEntry, V2BenchmarkReport,
    V2BenchmarkSurface, V2LeafUsageBin, V2ObservabilitySnapshot,
    DEFAULT_V2_BENCHMARK_SEQUENCE_LENGTHS,
};
pub use v2_fixture::{
    build_baseline_v2_synthetic_model, BaselineV2SyntheticModel, BaselineV2SyntheticModelConfig,
};
pub use v2_synthetic::{
    default_v2_synthetic_probe_suites, run_v2_synthetic_probe_suite, run_v2_synthetic_probe_suites,
    SyntheticProbeKind, SyntheticProbeMetrics, SyntheticProbeMode, SyntheticProbeModeReport,
    SyntheticProbeModel, SyntheticProbeReport, SyntheticProbeSample, SyntheticProbeSampleResult,
    SyntheticProbeSuite, SyntheticProbeSuiteReport,
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
