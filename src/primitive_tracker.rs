use crate::TournamentRunReport;

pub const TRACKER_PATH: &str = "docs/primitive-tracker.md";

pub fn primitive_tracker_reminder_lines(report: &TournamentRunReport) -> Vec<String> {
    let variant_names = report.variant_name_map();
    let mut lines = Vec::with_capacity(report.artifact.species.len() + 1);
    lines.push(format!(
        "primitive-tracker reminder [{}]: review and update {TRACKER_PATH}",
        report.comparison_label()
    ));
    lines.extend(report.artifact.species.iter().map(|record| {
        let variant_name = variant_names
            .get(&record.stage.species)
            .copied()
            .unwrap_or_else(|| record.stage.species.as_str());
        if let Some(result) = report
            .results
            .iter()
            .find(|result| result.species == record.stage.species)
        {
            format!(
                "  {} fitness={:.2} stability={:.2} perplexity={:.2} arc={:.2} tok/s={:.0}",
                variant_name,
                result.fitness,
                result.stability_score,
                result.long_context_perplexity,
                result.arc_accuracy,
                result.tokens_per_sec,
            )
        } else {
            format!(
                "  {} outcome={} error={}",
                variant_name,
                outcome_label(record.outcome_class()),
                record.error.as_deref().unwrap_or("unknown error"),
            )
        }
    }));
    lines
}

fn outcome_label(outcome: fractal_core::RunOutcomeClass) -> &'static str {
    match outcome {
        fractal_core::RunOutcomeClass::Success => "success",
        fractal_core::RunOutcomeClass::TrainTimeout => "train-timeout",
        fractal_core::RunOutcomeClass::EvalConstrained => "eval-constrained",
        fractal_core::RunOutcomeClass::NumericFailure => "numeric-failure",
        fractal_core::RunOutcomeClass::LowSignal => "low-signal",
        fractal_core::RunOutcomeClass::RuntimeCost => "runtime-cost",
        fractal_core::RunOutcomeClass::InfraFailure => "infra-failure",
    }
}

#[cfg(test)]
mod tests {
    use super::primitive_tracker_reminder_lines;
    use crate::{species_registry_for_species, ComparisonAuthority, TournamentRunReport};
    use fractal_core::{
        PhaseTiming, RankedSpeciesResult, RunExecutionOutcome, RunManifest, RunPhase,
        RunQualityOutcome, SpeciesId, SpeciesRunArtifact, SpeciesRunStage, TournamentRunArtifact,
    };

    #[test]
    fn tracker_reminder_uses_report_context_and_variant_names() {
        let report = TournamentRunReport::new(
            crate::TournamentPreset::BullpenPolish,
            crate::TournamentLane::Challenger,
            ComparisonAuthority::AuthoritativeSamePreset,
            crate::TournamentPreset::BullpenPolish.config(),
            species_registry_for_species(SpeciesId::P1FractalHybrid),
            vec![RankedSpeciesResult {
                rank: 1,
                species: SpeciesId::P1FractalHybrid,
                stability_score: 1.49,
                long_context_perplexity: 1.67,
                arc_accuracy: 0.67,
                tokens_per_sec: 22.0,
                fitness: 0.40,
            }],
            TournamentRunArtifact {
                config: crate::TournamentPreset::BullpenPolish.config(),
                species: vec![SpeciesRunArtifact {
                    stage: SpeciesRunStage {
                        species: SpeciesId::P1FractalHybrid,
                        ordinal: 1,
                        total: 1,
                    },
                    manifest: RunManifest {
                        variant_name: fractal_core::PrimitiveVariantName::new_unchecked(
                            "p1_fractal_hybrid_v1",
                        ),
                        timeout_budget: None,
                        config: crate::TournamentPreset::BullpenPolish.config(),
                    },
                    phase_timings: vec![PhaseTiming {
                        phase: RunPhase::Train,
                        elapsed: std::time::Duration::from_secs(1),
                        completed: 1,
                        total: 1,
                    }],
                    execution_outcome: RunExecutionOutcome::Success,
                    quality_outcome: RunQualityOutcome::Clean,
                    error: None,
                    metrics: None,
                }],
            },
        );
        let lines = primitive_tracker_reminder_lines(&report);

        assert_eq!(
            lines[0],
            "primitive-tracker reminder [authoritative same-preset]: review and update docs/primitive-tracker.md"
        );
        assert!(lines[1].contains("p1_fractal_hybrid_v1"));
    }
}
