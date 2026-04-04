use std::collections::BTreeMap;

use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::Serialize;

use fractal_core::error::FractalError;

use crate::v2_training::ensure_empty_output_dir;
use crate::{
    load_baseline_v2_checkpoint_model, run_baseline_v2_smoke_train,
    run_v2_synthetic_probe_suites_with_modes, BaselineV2SyntheticModelConfig, SyntheticProbeKind,
    SyntheticProbeMetrics, SyntheticProbeMode, SyntheticProbeReport, SyntheticProbeSuite,
    V2CheckpointKind, V2CheckpointSelection, V2RootTopology, V2SmokeTrainConfig,
    V2SmokeTrainReport,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V2RequiredAblationStep {
    SingleRootNoMemory,
    MultiRootNoMemory,
    SingleRootSummariesOnly,
    SingleRootSparseRetrieval,
    SingleRootSparseRetrievalPlusExactLeafRead,
    MultiRootSummariesOnly,
    MultiRootSparseRetrieval,
    MultiRootSparseRetrievalWithoutExactLeafRead,
    MultiRootSparseRetrievalPlusExactLeafRead,
}

impl V2RequiredAblationStep {
    pub const ALL: [Self; 9] = [
        Self::SingleRootNoMemory,
        Self::MultiRootNoMemory,
        Self::SingleRootSummariesOnly,
        Self::SingleRootSparseRetrieval,
        Self::SingleRootSparseRetrievalPlusExactLeafRead,
        Self::MultiRootSummariesOnly,
        Self::MultiRootSparseRetrieval,
        Self::MultiRootSparseRetrievalWithoutExactLeafRead,
        Self::MultiRootSparseRetrievalPlusExactLeafRead,
    ];

    pub const fn step_number(self) -> usize {
        match self {
            Self::SingleRootNoMemory => 1,
            Self::MultiRootNoMemory => 2,
            Self::SingleRootSummariesOnly => 3,
            Self::SingleRootSparseRetrieval => 4,
            Self::SingleRootSparseRetrievalPlusExactLeafRead => 5,
            Self::MultiRootSummariesOnly => 6,
            Self::MultiRootSparseRetrieval => 7,
            Self::MultiRootSparseRetrievalWithoutExactLeafRead => 8,
            Self::MultiRootSparseRetrievalPlusExactLeafRead => 9,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::SingleRootNoMemory => "single_root_no_memory",
            Self::MultiRootNoMemory => "multi_root_no_memory",
            Self::SingleRootSummariesOnly => "single_root_summaries_only",
            Self::SingleRootSparseRetrieval => "single_root_sparse_retrieval",
            Self::SingleRootSparseRetrievalPlusExactLeafRead => {
                "single_root_sparse_retrieval_plus_exact_leaf_read"
            }
            Self::MultiRootSummariesOnly => "multi_root_summaries_only",
            Self::MultiRootSparseRetrieval => "multi_root_sparse_retrieval",
            Self::MultiRootSparseRetrievalWithoutExactLeafRead => {
                "multi_root_sparse_retrieval_without_exact_leaf_read"
            }
            Self::MultiRootSparseRetrievalPlusExactLeafRead => {
                "multi_root_sparse_retrieval_plus_exact_leaf_read"
            }
        }
    }

    pub const fn topology(self) -> V2RootTopology {
        match self {
            Self::SingleRootNoMemory
            | Self::SingleRootSummariesOnly
            | Self::SingleRootSparseRetrieval
            | Self::SingleRootSparseRetrievalPlusExactLeafRead => V2RootTopology::SingleRoot,
            Self::MultiRootNoMemory
            | Self::MultiRootSummariesOnly
            | Self::MultiRootSparseRetrieval
            | Self::MultiRootSparseRetrievalWithoutExactLeafRead
            | Self::MultiRootSparseRetrievalPlusExactLeafRead => V2RootTopology::MultiRoot,
        }
    }

    pub const fn mode(self) -> SyntheticProbeMode {
        match self {
            Self::SingleRootNoMemory | Self::MultiRootNoMemory => SyntheticProbeMode::NoMemory,
            Self::SingleRootSummariesOnly | Self::MultiRootSummariesOnly => {
                SyntheticProbeMode::SummariesOnly
            }
            Self::SingleRootSparseRetrieval
            | Self::MultiRootSparseRetrieval
            | Self::MultiRootSparseRetrievalWithoutExactLeafRead => SyntheticProbeMode::TreeOnly,
            Self::SingleRootSparseRetrievalPlusExactLeafRead
            | Self::MultiRootSparseRetrievalPlusExactLeafRead => {
                SyntheticProbeMode::TreePlusExactRead
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LearnedAblationConfig {
    pub smoke: V2SmokeTrainConfig,
    pub checkpoint_selection: V2CheckpointSelection,
    pub suites: Vec<SyntheticProbeSuite>,
}

impl V2LearnedAblationConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.smoke.validate()?;
        if self.checkpoint_selection != V2CheckpointSelection::Final {
            return Err(FractalError::InvalidConfig(format!(
                "v2_learned_ablation.checkpoint_selection must be 'final' so both topologies are compared on trained checkpoints, got '{}'",
                self.checkpoint_selection.label()
            )));
        }
        if self.smoke.model.root_count < 2 {
            return Err(FractalError::InvalidConfig(format!(
                "v2_learned_ablation.smoke.model.root_count must be at least 2 for the multi-root control, got {}",
                self.smoke.model.root_count
            )));
        }
        if self.suites.is_empty() {
            return Err(FractalError::InvalidConfig(
                "v2_learned_ablation.suites must contain at least one suite".to_string(),
            ));
        }
        for suite in &self.suites {
            suite.validate_for_model(self.smoke.model.vocab_size, self.smoke.model.leaf_size)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LearnedAblationSuiteMetrics {
    pub kind: SyntheticProbeKind,
    pub sample_count: usize,
    pub metrics: SyntheticProbeMetrics,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LearnedAblationStepReport {
    pub step: V2RequiredAblationStep,
    pub topology: V2RootTopology,
    pub mode: SyntheticProbeMode,
    pub suite_reports: Vec<V2LearnedAblationSuiteMetrics>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LearnedAblationTopologyReport {
    pub topology: V2RootTopology,
    pub model_config: BaselineV2SyntheticModelConfig,
    pub checkpoint_selection: V2CheckpointSelection,
    pub evaluated_checkpoint_kind: V2CheckpointKind,
    pub smoke: V2SmokeTrainReport,
    pub probe: SyntheticProbeReport,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2LearnedAblationReport {
    pub note: String,
    pub config: V2LearnedAblationConfig,
    pub topology_runs: Vec<V2LearnedAblationTopologyReport>,
    pub ordered_steps: Vec<V2LearnedAblationStepReport>,
}

impl V2LearnedAblationReport {
    pub fn topology(&self, topology: V2RootTopology) -> Option<&V2LearnedAblationTopologyReport> {
        self.topology_runs
            .iter()
            .find(|report| report.topology == topology)
    }

    pub fn step(&self, step: V2RequiredAblationStep) -> Option<&V2LearnedAblationStepReport> {
        self.ordered_steps.iter().find(|report| report.step == step)
    }
}

pub fn run_required_v2_learned_ablation_matrix<TrainB, EvalB>(
    config: V2LearnedAblationConfig,
    train_device: &TrainB::Device,
    eval_device: &EvalB::Device,
) -> Result<V2LearnedAblationReport, FractalError>
where
    TrainB: AutodiffBackend,
    EvalB: Backend,
{
    config.validate()?;
    ensure_empty_output_dir(&config.smoke.output_dir)?;
    let mut topology_runs = Vec::with_capacity(V2RootTopology::ALL.len());

    for topology in V2RootTopology::ALL {
        let mut topology_smoke = config.smoke.clone();
        topology_smoke.output_dir = topology_smoke
            .output_dir
            .join(topology.output_directory_name());
        topology_smoke.model = topology_smoke
            .model
            .with_root_count_preserving_total_budget(topology.root_count(&config.smoke.model));

        let training = run_baseline_v2_smoke_train::<TrainB>(topology_smoke, train_device)?;
        let evaluated_checkpoint_kind = config.checkpoint_selection.resolved_kind(&training.report);
        let loaded = load_baseline_v2_checkpoint_model::<EvalB>(
            &training.report.checkpoint.report_path,
            config.checkpoint_selection,
            eval_device,
        )?;
        let probe = run_v2_synthetic_probe_suites_with_modes(
            &loaded.model,
            &config.suites,
            &required_v2_learned_ablation_modes(),
            eval_device,
        )?;

        topology_runs.push(V2LearnedAblationTopologyReport {
            topology,
            model_config: loaded.report.config.model,
            checkpoint_selection: loaded.selection,
            evaluated_checkpoint_kind,
            smoke: training.report,
            probe,
        });
    }

    let ordered_steps = build_ordered_step_reports(&topology_runs)?;

    Ok(V2LearnedAblationReport {
        note: format!(
            "single-root and multi-root checkpoints are trained separately at equal total root state/readout budget, then evaluated from the {} checkpoint across no-memory, summaries-only, tree-only, and tree-plus-exact-read probe modes; checklist step 7 preserves the original sparse-retrieval wording and aliases the tree-only retrieval case",
            config.checkpoint_selection.label()
        ),
        config,
        topology_runs,
        ordered_steps,
    })
}

pub const fn required_v2_learned_ablation_modes() -> [SyntheticProbeMode; 4] {
    [
        SyntheticProbeMode::NoMemory,
        SyntheticProbeMode::SummariesOnly,
        SyntheticProbeMode::TreeOnly,
        SyntheticProbeMode::TreePlusExactRead,
    ]
}

fn build_ordered_step_reports(
    topology_runs: &[V2LearnedAblationTopologyReport],
) -> Result<Vec<V2LearnedAblationStepReport>, FractalError> {
    let by_topology = topology_runs
        .iter()
        .map(|report| (report.topology, report))
        .collect::<BTreeMap<_, _>>();
    let mut ordered_steps = Vec::with_capacity(V2RequiredAblationStep::ALL.len());

    for step in V2RequiredAblationStep::ALL {
        let topology_report = by_topology.get(&step.topology()).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "v2_learned_ablation missing topology report for {:?}",
                step.topology()
            ))
        })?;
        ordered_steps.push(V2LearnedAblationStepReport {
            step,
            topology: step.topology(),
            mode: step.mode(),
            suite_reports: suite_metrics_for_mode(&topology_report.probe, step.mode())?,
        });
    }

    Ok(ordered_steps)
}

fn suite_metrics_for_mode(
    report: &SyntheticProbeReport,
    mode: SyntheticProbeMode,
) -> Result<Vec<V2LearnedAblationSuiteMetrics>, FractalError> {
    let mut suites = Vec::with_capacity(report.suites.len());
    for suite in &report.suites {
        let mode_report = suite.mode_report(mode).ok_or_else(|| {
            FractalError::InvalidState(format!(
                "v2_learned_ablation suite {:?} is missing mode {:?}",
                suite.kind, mode
            ))
        })?;
        suites.push(V2LearnedAblationSuiteMetrics {
            kind: suite.kind,
            sample_count: suite.sample_count,
            metrics: mode_report.metrics,
        });
    }
    Ok(suites)
}

impl V2RootTopology {
    const fn root_count(self, model: &BaselineV2SyntheticModelConfig) -> usize {
        match self {
            Self::SingleRoot => 1,
            Self::MultiRoot => model.root_count,
        }
    }

    const fn output_directory_name(self) -> &'static str {
        match self {
            Self::SingleRoot => "single-root-smoke",
            Self::MultiRoot => "multi-root-smoke",
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::{backend::Autodiff, backend::Candle};

    use super::*;
    use crate::{default_v2_synthetic_probe_suites, filter_synthetic_probe_suites};

    type TrainBackend = Autodiff<Candle<f32, i64>>;
    type EvalBackend = Candle<f32, i64>;

    #[test]
    fn learned_ablation_config_rejects_single_root_base_model() {
        let root = unique_temp_dir("v2-learned-ablation-invalid");
        let config = V2LearnedAblationConfig {
            smoke: V2SmokeTrainConfig::new(vec![root.join("corpus.md")], root.join("artifacts")),
            checkpoint_selection: V2CheckpointSelection::Final,
            suites: default_v2_synthetic_probe_suites(),
        };
        let invalid = V2LearnedAblationConfig {
            smoke: V2SmokeTrainConfig {
                model: config
                    .smoke
                    .model
                    .with_root_count_preserving_total_budget(1),
                ..config.smoke
            },
            checkpoint_selection: V2CheckpointSelection::Final,
            suites: config.suites,
        };

        let error = invalid.validate().unwrap_err();
        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("root_count")
        ));
    }

    #[test]
    fn learned_ablation_matrix_trains_both_topologies_and_builds_ordered_steps() {
        let root = unique_temp_dir("v2-learned-ablation");
        let corpus_path = root.join("corpus.md");
        fs::write(
            &corpus_path,
            "v2 learned ablation smoke corpus\n".repeat(64),
        )
        .unwrap();
        let suites = filter_synthetic_probe_suites(
            default_v2_synthetic_probe_suites(),
            &[SyntheticProbeKind::Copy],
        );
        let mut smoke = V2SmokeTrainConfig::new(vec![corpus_path], root.join("artifacts"));
        smoke.train_steps = 1;
        smoke.eval_batches = 1;
        smoke.eval_holdout_every = 2;
        let config = V2LearnedAblationConfig {
            smoke,
            checkpoint_selection: V2CheckpointSelection::Final,
            suites,
        };

        let train_device = <TrainBackend as Backend>::Device::default();
        let eval_device = <EvalBackend as Backend>::Device::default();
        let report = run_required_v2_learned_ablation_matrix::<TrainBackend, EvalBackend>(
            config,
            &train_device,
            &eval_device,
        )
        .unwrap();

        assert_eq!(report.topology_runs.len(), 2);
        assert_eq!(
            report.ordered_steps.len(),
            V2RequiredAblationStep::ALL.len()
        );
        let single = report.topology(V2RootTopology::SingleRoot).unwrap();
        let multi = report.topology(V2RootTopology::MultiRoot).unwrap();
        assert_eq!(single.model_config.root_count, 1);
        assert_eq!(multi.model_config.root_count, 2);
        assert_eq!(
            single.model_config.total_root_state_dim,
            multi.model_config.total_root_state_dim
        );
        assert_eq!(
            single.model_config.total_root_readout_dim,
            multi.model_config.total_root_readout_dim
        );
        assert_eq!(single.checkpoint_selection, V2CheckpointSelection::Final);
        assert_eq!(
            single.evaluated_checkpoint_kind,
            V2CheckpointKind::FinalEval
        );
        assert_eq!(multi.checkpoint_selection, V2CheckpointSelection::Final);
        assert_eq!(multi.evaluated_checkpoint_kind, V2CheckpointKind::FinalEval);
        let step7 = report
            .step(V2RequiredAblationStep::MultiRootSparseRetrieval)
            .unwrap();
        let step8 = report
            .step(V2RequiredAblationStep::MultiRootSparseRetrievalWithoutExactLeafRead)
            .unwrap();
        assert_eq!(step7.mode, SyntheticProbeMode::TreeOnly);
        assert_eq!(step8.mode, SyntheticProbeMode::TreeOnly);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn learned_ablation_config_rejects_best_checkpoint_selection() {
        let root = unique_temp_dir("v2-learned-ablation-best");
        let config = V2LearnedAblationConfig {
            smoke: V2SmokeTrainConfig::new(vec![root.join("corpus.md")], root.join("artifacts")),
            checkpoint_selection: V2CheckpointSelection::Best,
            suites: default_v2_synthetic_probe_suites(),
        };

        let error = config.validate().unwrap_err();
        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("checkpoint_selection")
        ));
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "{prefix}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
