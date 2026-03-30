pub use fractal_core::*;
pub use fractal_eval_private::{aggregate_results, perplexity_score, speed_score, stability_score};
pub use fractal_primitives_private::{
    species_registry, B1FractalGated, B2StableHierarchical, B3FractalHierarchical, B4Universal,
    P1Contractive, P2Mandelbrot, P3Hierarchical, SPECIES_REGISTRY,
};

pub fn run_ranked_generation(
    tournament: &Tournament,
    species: &[SpeciesDefinition],
) -> Result<Vec<RankedSpeciesResult>, error::FractalError> {
    Ok(aggregate_results(tournament.run_generation(species)?))
}
