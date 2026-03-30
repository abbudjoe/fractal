use crate::registry::SpeciesId;

#[derive(Clone, Debug)]
pub struct SpeciesRawMetrics {
    pub species: SpeciesId,
    pub grad_norm_depth_20: f64,
    pub long_context_perplexity: f64,
    pub arc_accuracy: f64,
    pub tokens_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct RankedSpeciesResult {
    pub rank: usize,
    pub species: SpeciesId,
    pub stability_score: f64,
    pub long_context_perplexity: f64,
    pub arc_accuracy: f64,
    pub tokens_per_sec: f64,
    pub fitness: f64,
}
