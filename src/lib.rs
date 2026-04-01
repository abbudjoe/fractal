use std::collections::BTreeMap;

mod primitive_tracker;
mod run_artifacts;
mod tokenizer_training;

pub use fractal_core::*;
pub use fractal_eval_private::{aggregate_results, perplexity_score, speed_score, stability_score};
pub use fractal_primitives_private::{
    species_registry, B1FractalGated, B2StableHierarchical, B3FractalHierarchical, B4Universal,
    GeneralizedMobius, Ifs, JuliaRecursiveEscape, LogisticChaoticMap,
    MandelboxRecursiveDynEscapeRadius, P1Contractive, P1FractalHybrid, P1FractalHybridComposite,
    P1FractalHybridDynGate, P2Mandelbrot, P3Hierarchical, SPECIES_REGISTRY,
};
pub use primitive_tracker::{primitive_tracker_reminder_lines, TRACKER_PATH};
pub use run_artifacts::{persist_run_artifacts, PersistedRunPaths};
pub use tokenizer_training::{
    load_stage0_tokenizer_runtime, materialize_bridge_vocab_artifact, run_tokenizer_backed_species,
    run_tokenizer_backed_species_from_experiment, ResolvedTokenizerArtifact, Stage0PadSemantics,
    Stage0SlowTokenizer, TextCorpusSplitSource, TokenizerBridgeStats, TokenizerTrainingCorpus,
    TokenizerTrainingCorpusSource, TokenizerTrainingRuntime, STAGE0_CANONICAL_TOKENIZER_FILENAME,
    STAGE0_CANONICAL_TOKENIZER_REPO_ID, STAGE0_CANONICAL_TOKENIZER_USE_FAST,
};

#[derive(Clone)]
pub struct TournamentRunReport {
    pub preset: TournamentPreset,
    pub lane: TournamentLane,
    pub comparison: ComparisonContract,
    pub config: TournamentConfig,
    pub species: Vec<SpeciesDefinition>,
    pub results: Vec<RankedSpeciesResult>,
    pub artifact: TournamentRunArtifact,
    pub bridge_stats: BTreeMap<SpeciesId, TokenizerBridgeStats>,
}

#[derive(Clone)]
pub struct TournamentRunReportParts {
    pub preset: TournamentPreset,
    pub lane: TournamentLane,
    pub comparison: ComparisonContract,
    pub config: TournamentConfig,
    pub species: Vec<SpeciesDefinition>,
    pub results: Vec<RankedSpeciesResult>,
    pub artifact: TournamentRunArtifact,
    pub bridge_stats: BTreeMap<SpeciesId, TokenizerBridgeStats>,
}

impl TournamentRunReport {
    pub fn new(parts: TournamentRunReportParts) -> Self {
        Self {
            preset: parts.preset,
            lane: parts.lane,
            comparison: parts.comparison,
            config: parts.config,
            species: parts.species,
            results: parts.results,
            artifact: parts.artifact,
            bridge_stats: parts.bridge_stats,
        }
    }

    pub fn comparison_label(&self) -> &'static str {
        self.comparison.label()
    }

    pub fn runtime_surface_label(&self) -> String {
        self.artifact
            .species
            .first()
            .and_then(|record| record.manifest.experiment.as_ref())
            .map(|experiment| experiment.runtime.label())
            .unwrap_or_else(|| RuntimeSurfaceSpec::default().label())
    }

    pub fn variant_name_for(&self, species: fractal_core::SpeciesId) -> &'static str {
        self.species
            .iter()
            .find(|definition| definition.id == species)
            .map(|definition| definition.variant_name.as_str())
            .unwrap_or_else(|| species.as_str())
    }

    pub fn variant_name_map(&self) -> BTreeMap<fractal_core::SpeciesId, &'static str> {
        self.species
            .iter()
            .map(|definition| (definition.id, definition.variant_name.as_str()))
            .collect()
    }
}

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
    use fractal_core::lifecycle::TrainingRuntimeArtifact;
    use fractal_core::{
        ComparisonContract, PhaseTiming, RunExecutionOutcome, RunManifest, RunPhase,
        RunQualityOutcome, SpeciesRunArtifact, SpeciesRunStage, TournamentRunArtifact,
    };

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

    #[test]
    fn run_report_marks_same_preset_and_mixed_preset_comparisons() {
        let species = species_registry_for_species(fractal_core::SpeciesId::P1Contractive);
        let report = TournamentRunReport::new(TournamentRunReportParts {
            preset: TournamentPreset::GenerationFour,
            lane: TournamentLane::Leader,
            comparison: ComparisonContract::authoritative_same_preset(),
            config: TournamentPreset::GenerationFour.config(),
            species,
            results: vec![RankedSpeciesResult {
                rank: 1,
                species: fractal_core::SpeciesId::P1Contractive,
                stability_score: 0.53,
                long_context_perplexity: 1.54,
                arc_accuracy: 0.68,
                tokens_per_sec: 114.0,
                fitness: 0.58,
            }],
            artifact: single_species_artifact(
                fractal_core::SpeciesId::P1Contractive,
                "p1_contractive_v1",
            ),
            bridge_stats: BTreeMap::new(),
        });

        assert!(report.comparison.is_authoritative_same_preset());
        assert_eq!(report.comparison_label(), "authoritative same-preset");
        assert_eq!(report.runtime_surface_label(), "conservative-defaults");
        assert_eq!(
            report.variant_name_for(fractal_core::SpeciesId::P1Contractive),
            "p1_contractive_v1"
        );

        let advisory = TournamentRunReport::new(TournamentRunReportParts {
            preset: TournamentPreset::FastTest,
            lane: TournamentLane::Baseline,
            comparison: ComparisonContract::advisory_mixed_preset(),
            config: TournamentPreset::FastTest.config(),
            species: species_registry_for_species(fractal_core::SpeciesId::P3Hierarchical),
            results: vec![RankedSpeciesResult {
                rank: 1,
                species: fractal_core::SpeciesId::P3Hierarchical,
                stability_score: 0.55,
                long_context_perplexity: 1.66,
                arc_accuracy: 0.44,
                tokens_per_sec: 34.0,
                fitness: 0.58,
            }],
            artifact: single_species_artifact(
                fractal_core::SpeciesId::P3Hierarchical,
                "p3_hierarchical_v1",
            ),
            bridge_stats: BTreeMap::new(),
        });

        assert_eq!(advisory.comparison_label(), "advisory mixed-preset");
        assert_eq!(
            advisory.variant_name_for(fractal_core::SpeciesId::P3Hierarchical),
            "p3_hierarchical_v1"
        );
    }

    fn single_species_artifact(
        species: fractal_core::SpeciesId,
        variant_name: &'static str,
    ) -> TournamentRunArtifact {
        TournamentRunArtifact {
            config: TournamentPreset::FastTest.config(),
            species: vec![SpeciesRunArtifact {
                stage: SpeciesRunStage {
                    species,
                    ordinal: 1,
                    total: 1,
                },
                manifest: RunManifest {
                    variant_name: fractal_core::PrimitiveVariantName::new_unchecked(variant_name),
                    timeout_budget: None,
                    config: TournamentPreset::FastTest.config(),
                    experiment: None,
                },
                phase_timings: vec![PhaseTiming {
                    phase: RunPhase::Train,
                    elapsed: std::time::Duration::from_secs(1),
                    completed: 1,
                    total: 1,
                }],
                training_runtime: TrainingRuntimeArtifact::default(),
                execution_outcome: RunExecutionOutcome::Success,
                quality_outcome: RunQualityOutcome::Clean,
                error: None,
                metrics: None,
            }],
        }
    }
}
