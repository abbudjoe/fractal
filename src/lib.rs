pub use fractal_core::*;
pub use fractal_eval_private::{aggregate_results, perplexity_score, speed_score, stability_score};
pub use fractal_primitives_private::{
    species_registry, B1FractalGated, B2StableHierarchical, B3FractalHierarchical, B4Universal,
    GeneralizedMobius, Ifs, LogisticChaoticMap, P1Contractive, P1FractalHybrid, P2Mandelbrot,
    P3Hierarchical, SPECIES_REGISTRY,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TournamentLane {
    All,
    Baseline,
    Challenger,
    ProvingGround,
    Leader,
}

const CHALLENGER_SPECIES: [fractal_core::SpeciesId; 3] = [
    fractal_core::SpeciesId::Ifs,
    fractal_core::SpeciesId::GeneralizedMobius,
    fractal_core::SpeciesId::LogisticChaoticMap,
];
const PROVING_GROUND_SPECIES: [fractal_core::SpeciesId; 5] = [
    fractal_core::SpeciesId::B1FractalGated,
    fractal_core::SpeciesId::P1FractalHybrid,
    fractal_core::SpeciesId::P2Mandelbrot,
    fractal_core::SpeciesId::B3FractalHierarchical,
    fractal_core::SpeciesId::B4Universal,
];

const LEADER_SPECIES: [fractal_core::SpeciesId; 1] = [fractal_core::SpeciesId::P1Contractive];

impl TournamentLane {
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Baseline => "baseline",
            Self::Challenger => "challenger",
            Self::ProvingGround => "proving-ground",
            Self::Leader => "leader",
        }
    }

    pub fn default_preset(self) -> TournamentPreset {
        match self {
            Self::All => TournamentPreset::Default,
            Self::Baseline => TournamentPreset::ResearchMedium,
            Self::Challenger => TournamentPreset::BullpenPolish,
            Self::ProvingGround => TournamentPreset::MinimalProvingGround,
            Self::Leader => TournamentPreset::GenerationFour,
        }
    }
}

pub fn species_registry_for_lane(lane: TournamentLane) -> Vec<SpeciesDefinition> {
    match lane {
        TournamentLane::All | TournamentLane::Baseline => species_registry().to_vec(),
        TournamentLane::Challenger => filter_species_registry(&CHALLENGER_SPECIES),
        TournamentLane::ProvingGround => filter_species_registry(&PROVING_GROUND_SPECIES),
        TournamentLane::Leader => filter_species_registry(&LEADER_SPECIES),
    }
}

pub fn species_registry_for_species(species: fractal_core::SpeciesId) -> Vec<SpeciesDefinition> {
    filter_species_registry(&[species])
}

fn filter_species_registry(ids: &[fractal_core::SpeciesId]) -> Vec<SpeciesDefinition> {
    species_registry()
        .iter()
        .copied()
        .filter(|definition| ids.contains(&definition.id))
        .collect()
}

pub fn run_ranked_generation(
    tournament: &Tournament,
    species: &[SpeciesDefinition],
) -> Result<Vec<RankedSpeciesResult>, error::FractalError> {
    Ok(aggregate_results(tournament.run_generation(species)?))
}

pub fn run_ranked_generation_with_reporter(
    tournament: &Tournament,
    species: &[SpeciesDefinition],
    reporter: std::sync::Arc<dyn TournamentReporter>,
) -> Result<Vec<RankedSpeciesResult>, error::FractalError> {
    Ok(aggregate_results(
        tournament.run_generation_with_reporter(species, Some(reporter))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenger_lane_only_includes_bullpen_species() {
        let ids: Vec<_> = species_registry_for_lane(TournamentLane::Challenger)
            .into_iter()
            .map(|definition| definition.id)
            .collect();

        assert_eq!(ids, CHALLENGER_SPECIES);
    }

    #[test]
    fn proving_ground_lane_only_includes_squaring_species() {
        let ids: Vec<_> = species_registry_for_lane(TournamentLane::ProvingGround)
            .into_iter()
            .map(|definition| definition.id)
            .collect();

        assert_eq!(ids, PROVING_GROUND_SPECIES);
    }

    #[test]
    fn leader_lane_only_includes_current_leader() {
        let ids: Vec<_> = species_registry_for_lane(TournamentLane::Leader)
            .into_iter()
            .map(|definition| definition.id)
            .collect();

        assert_eq!(ids, LEADER_SPECIES);
    }

    #[test]
    fn single_species_registry_only_includes_requested_candidate() {
        let ids: Vec<_> = species_registry_for_species(fractal_core::SpeciesId::GeneralizedMobius)
            .into_iter()
            .map(|definition| definition.id)
            .collect();

        assert_eq!(ids, [fractal_core::SpeciesId::GeneralizedMobius]);
    }
}
