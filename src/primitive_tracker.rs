use std::collections::BTreeMap;

use fractal_core::{RankedSpeciesResult, SpeciesDefinition, SpeciesId};

pub const TRACKER_PATH: &str = "docs/primitive-tracker.md";

pub fn primitive_tracker_reminder_lines(
    results: &[RankedSpeciesResult],
    species: &[SpeciesDefinition],
) -> Vec<String> {
    let variant_names = species
        .iter()
        .map(|definition| (definition.id, definition.variant_name.as_str()))
        .collect::<BTreeMap<SpeciesId, &'static str>>();
    let mut lines = Vec::with_capacity(results.len() + 1);
    lines.push(format!(
        "primitive-tracker reminder: review and update {TRACKER_PATH}"
    ));
    lines.extend(results.iter().map(|result| {
        let variant_name = variant_names
            .get(&result.species)
            .copied()
            .unwrap_or_else(|| result.species.as_str());
        format!(
            "  {} fitness={:.2} stability={:.2} perplexity={:.2} arc={:.2} tok/s={:.0}",
            variant_name,
            result.fitness,
            result.stability_score,
            result.long_context_perplexity,
            result.arc_accuracy,
            result.tokens_per_sec,
        )
    }));
    lines
}

#[cfg(test)]
mod tests {
    use super::primitive_tracker_reminder_lines;
    use crate::species_registry_for_species;
    use fractal_core::{RankedSpeciesResult, SpeciesId};

    #[test]
    fn tracker_reminder_uses_registered_variant_names() {
        let species = species_registry_for_species(SpeciesId::P1FractalHybrid);
        let lines = primitive_tracker_reminder_lines(
            &[RankedSpeciesResult {
                rank: 1,
                species: SpeciesId::P1FractalHybrid,
                stability_score: 1.49,
                long_context_perplexity: 1.67,
                arc_accuracy: 0.67,
                tokens_per_sec: 22.0,
                fitness: 0.40,
            }],
            &species,
        );

        assert_eq!(
            lines[0],
            "primitive-tracker reminder: review and update docs/primitive-tracker.md"
        );
        assert!(lines[1].contains("p1_fractal_hybrid_v1"));
    }
}
