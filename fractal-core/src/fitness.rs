#[derive(Clone, Debug)]
pub struct SpeciesRawMetrics {
    pub species: String,
    pub grad_norm_depth_20: f64,
    pub long_context_perplexity: f64,
    pub arc_accuracy: f64,
    pub tokens_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct RankedSpeciesResult {
    pub rank: usize,
    pub species: String,
    pub stability_score: f64,
    pub long_context_perplexity: f64,
    pub arc_accuracy: f64,
    pub tokens_per_sec: f64,
    pub fitness: f64,
}

pub fn stability_score(grad_norm_depth_20: f64) -> f64 {
    if !grad_norm_depth_20.is_finite() {
        0.0
    } else {
        1.0 / (1.0 + grad_norm_depth_20)
    }
}

pub fn perplexity_score(perplexity: f64) -> f64 {
    1.0 / perplexity.max(1.0)
}

pub fn speed_score(tokens_per_sec: f64, best_tokens_per_sec: f64) -> f64 {
    if best_tokens_per_sec <= f64::EPSILON {
        0.0
    } else {
        tokens_per_sec / best_tokens_per_sec
    }
}

pub fn aggregate_results(mut metrics: Vec<SpeciesRawMetrics>) -> Vec<RankedSpeciesResult> {
    let best_tokens = metrics
        .iter()
        .map(|metric| metric.tokens_per_sec)
        .fold(0.0f64, f64::max);

    let mut ranked = metrics
        .drain(..)
        .map(|metric| {
            let stability = stability_score(metric.grad_norm_depth_20);
            let perplexity_component = perplexity_score(metric.long_context_perplexity);
            let speed = speed_score(metric.tokens_per_sec, best_tokens);
            let fitness = 0.35 * stability
                + 0.30 * perplexity_component
                + 0.25 * metric.arc_accuracy
                + 0.10 * speed;

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
