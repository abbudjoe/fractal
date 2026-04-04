use burn::tensor::backend::Backend;
use serde::Serialize;

use fractal_core::error::FractalError;

use crate::{
    build_baseline_v2_synthetic_model, run_v2_synthetic_probe_suites,
    BaselineV2SyntheticModelConfig, SyntheticProbeKind, SyntheticProbeMode, SyntheticProbeReport,
    SyntheticProbeSuite,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum V2RootTopology {
    SingleRoot,
    MultiRoot,
}

impl V2RootTopology {
    pub const ALL: [Self; 2] = [Self::SingleRoot, Self::MultiRoot];

    pub const fn root_count(self) -> usize {
        match self {
            Self::SingleRoot => 1,
            Self::MultiRoot => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Default)]
pub struct V2AblationConfig {
    pub base_model: BaselineV2SyntheticModelConfig,
}

impl V2AblationConfig {
    pub fn validate(&self) -> Result<(), FractalError> {
        self.base_model.validate()?;
        if self.base_model.root_count < 2 {
            return Err(FractalError::InvalidConfig(format!(
                "v2_ablation.base_model.root_count must be at least 2 for the multi-root control, got {}",
                self.base_model.root_count
            )));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2AblationCaseReport {
    pub topology: V2RootTopology,
    pub model_config: BaselineV2SyntheticModelConfig,
    pub synthetic: SyntheticProbeReport,
}

impl V2AblationCaseReport {
    pub fn suite(&self, kind: SyntheticProbeKind) -> Option<&crate::SyntheticProbeSuiteReport> {
        self.synthetic
            .suites
            .iter()
            .find(|suite| suite.kind == kind)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct V2AblationReport {
    pub note: String,
    pub cases: Vec<V2AblationCaseReport>,
}

impl V2AblationReport {
    pub fn case(&self, topology: V2RootTopology) -> Option<&V2AblationCaseReport> {
        self.cases.iter().find(|case| case.topology == topology)
    }
}

pub fn run_required_v2_ablation_sweep<B: Backend>(
    config: V2AblationConfig,
    suites: &[SyntheticProbeSuite],
    device: &B::Device,
) -> Result<V2AblationReport, FractalError> {
    config.validate()?;
    let mut cases = Vec::with_capacity(V2RootTopology::ALL.len());

    for topology in V2RootTopology::ALL {
        let case_config = config
            .base_model
            .with_root_count_preserving_total_budget(topology.root_count());
        let model = build_baseline_v2_synthetic_model::<B>(case_config, device)?;
        let synthetic = run_v2_synthetic_probe_suites(&model, suites, device)?;
        cases.push(V2AblationCaseReport {
            topology,
            model_config: case_config,
            synthetic,
        });
    }

    Ok(V2AblationReport {
        note: "single-root and multi-root cases preserve total root state and readout budget; compare no-memory, summaries-only, tree-only, and tree-plus-exact-read modes within each topology".to_string(),
        cases,
    })
}

pub fn required_v2_ablation_modes() -> [SyntheticProbeMode; 4] {
    [
        SyntheticProbeMode::NoMemory,
        SyntheticProbeMode::SummariesOnly,
        SyntheticProbeMode::TreeOnly,
        SyntheticProbeMode::TreePlusExactRead,
    ]
}

#[cfg(test)]
mod tests {
    use burn::backend::Candle;

    use super::*;
    use crate::default_v2_synthetic_probe_suites;

    type TestBackend = Candle<f32, i64>;

    #[test]
    fn ablation_config_rejects_single_root_base_model() {
        let error = V2AblationConfig {
            base_model: BaselineV2SyntheticModelConfig::default()
                .with_root_count_preserving_total_budget(1),
        }
        .validate()
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("root_count")
        ));
    }

    #[test]
    fn required_ablation_sweep_runs_with_equal_root_budgets() {
        let device = Default::default();
        let report = run_required_v2_ablation_sweep::<TestBackend>(
            V2AblationConfig::default(),
            &default_v2_synthetic_probe_suites(),
            &device,
        )
        .unwrap();

        assert_eq!(report.cases.len(), V2RootTopology::ALL.len());
        let single_root = report.case(V2RootTopology::SingleRoot).unwrap();
        let multi_root = report.case(V2RootTopology::MultiRoot).unwrap();

        assert_eq!(single_root.model_config.root_count, 1);
        assert_eq!(multi_root.model_config.root_count, 2);
        assert_eq!(
            single_root.model_config.total_root_state_dim,
            multi_root.model_config.total_root_state_dim
        );
        assert_eq!(
            single_root.model_config.total_root_readout_dim,
            multi_root.model_config.total_root_readout_dim
        );
        assert!(single_root
            .synthetic
            .suites
            .iter()
            .all(|suite| suite.mode_reports.len() == required_v2_ablation_modes().len()));
    }
}
