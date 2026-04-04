use std::collections::BTreeMap;

use crate::error::FractalError;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalMemoryTaskFamily {
    OrdinaryLm,
    Copy,
    AssociativeRecall,
    Induction,
    NoisyRetrieval,
    Custom(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalMemoryIntervention {
    NoTreeRead,
    NoExactLeafRead,
    NextBestSpanSubstitution,
    RootDrop { root_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalMemoryAuditSample {
    pub batch_index: usize,
    pub position: usize,
    pub task_family: CausalMemoryTaskFamily,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalMemoryAuditPlan {
    samples: Vec<CausalMemoryAuditSample>,
    include_no_tree_read: bool,
    include_no_exact_leaf_read: bool,
    include_next_best_span_substitution: bool,
    include_root_drop: bool,
}

impl CausalMemoryAuditPlan {
    pub fn all(samples: Vec<CausalMemoryAuditSample>) -> Result<Self, FractalError> {
        let plan = Self {
            samples,
            include_no_tree_read: true,
            include_no_exact_leaf_read: true,
            include_next_best_span_substitution: true,
            include_root_drop: true,
        };
        plan.validate()
    }

    pub fn samples(&self) -> &[CausalMemoryAuditSample] {
        &self.samples
    }

    pub fn include_no_tree_read(&self) -> bool {
        self.include_no_tree_read
    }

    pub fn include_no_exact_leaf_read(&self) -> bool {
        self.include_no_exact_leaf_read
    }

    pub fn include_next_best_span_substitution(&self) -> bool {
        self.include_next_best_span_substitution
    }

    pub fn include_root_drop(&self) -> bool {
        self.include_root_drop
    }

    pub(crate) fn validate(self) -> Result<Self, FractalError> {
        if self.samples.is_empty() {
            return Err(FractalError::InvalidConfig(
                "causal_memory_audit.samples must not be empty".to_string(),
            ));
        }
        let mut seen = BTreeMap::new();
        for sample in &self.samples {
            let key = (
                sample.batch_index,
                sample.position,
                sample.task_family.clone(),
            );
            if seen.insert(key, ()).is_some() {
                return Err(FractalError::InvalidConfig(format!(
                    "causal_memory_audit contains a duplicate sample for batch {} position {} task {:?}",
                    sample.batch_index, sample.position, sample.task_family
                )));
            }
        }
        if !self.include_no_tree_read
            && !self.include_no_exact_leaf_read
            && !self.include_next_best_span_substitution
            && !self.include_root_drop
        {
            return Err(FractalError::InvalidConfig(
                "causal_memory_audit must enable at least one intervention".to_string(),
            ));
        }

        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalMemoryEvaluationContext {
    pub routing_depth: usize,
    pub span_distance: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CausalMemoryHeadContext {
    pub head_index: usize,
    pub routing_depth: usize,
    pub span_distances: Vec<usize>,
    pub selected_leaf_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalMemoryDeltaMetrics {
    pub loss_delta: f32,
    pub target_logit_delta: f32,
    pub kl_divergence: f32,
    pub retrieval_accuracy_delta: Option<f32>,
    pub perplexity_delta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryInterventionResult {
    pub intervention: CausalMemoryIntervention,
    pub applied: bool,
    pub context: Option<CausalMemoryEvaluationContext>,
    pub head_contexts: Vec<CausalMemoryHeadContext>,
    pub metrics: Option<CausalMemoryDeltaMetrics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryAuditSampleReport {
    pub sample: CausalMemoryAuditSample,
    pub reference_context: CausalMemoryEvaluationContext,
    pub reference_head_contexts: Vec<CausalMemoryHeadContext>,
    pub target_token_id: i64,
    pub reference_loss: f32,
    pub reference_target_logit: f32,
    pub interventions: Vec<CausalMemoryInterventionResult>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalMemoryAggregateStats {
    pub count: usize,
    pub average_loss_delta: f32,
    pub average_target_logit_delta: f32,
    pub average_kl_divergence: f32,
    pub retrieval_accuracy_count: usize,
    pub average_retrieval_accuracy_delta: Option<f32>,
    pub average_perplexity_delta: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryInterventionAggregate {
    pub intervention: CausalMemoryIntervention,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CausalMemoryComponentFamily {
    LocalTrunk,
    TreeSummaryRetrieval,
    ExactLeafRead,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryComponentFamilyAggregate {
    pub component_family: CausalMemoryComponentFamily,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryRootAggregate {
    pub root_index: usize,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryRoutingDepthAggregate {
    pub routing_depth: usize,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryRoutingHeadAggregate {
    pub head_index: usize,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemorySpanDistanceAggregate {
    pub span_distance: usize,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemorySelectedLeafAggregate {
    pub selected_leaf_index: usize,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryTaskFamilyAggregate {
    pub task_family: CausalMemoryTaskFamily,
    pub stats: CausalMemoryAggregateStats,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CausalMemoryAuditReport {
    pub sample_reports: Vec<CausalMemoryAuditSampleReport>,
    pub tree_retrieval_utility: Option<CausalMemoryAggregateStats>,
    pub exact_leaf_read_utility: Option<CausalMemoryAggregateStats>,
    pub utility_by_intervention: Vec<CausalMemoryInterventionAggregate>,
    pub utility_by_component_family: Vec<CausalMemoryComponentFamilyAggregate>,
    pub utility_by_root: Vec<CausalMemoryRootAggregate>,
    pub utility_by_routing_depth: Vec<CausalMemoryRoutingDepthAggregate>,
    pub utility_by_routing_head: Vec<CausalMemoryRoutingHeadAggregate>,
    pub utility_by_span_distance: Vec<CausalMemorySpanDistanceAggregate>,
    pub utility_by_selected_leaf: Vec<CausalMemorySelectedLeafAggregate>,
    pub utility_by_task_family: Vec<CausalMemoryTaskFamilyAggregate>,
}

impl CausalMemoryAuditReport {
    pub(crate) fn from_sample_reports(sample_reports: Vec<CausalMemoryAuditSampleReport>) -> Self {
        let mut intervention_groups: BTreeMap<
            CausalMemoryIntervention,
            Vec<CausalMemoryDeltaMetrics>,
        > = BTreeMap::new();
        let mut component_groups: BTreeMap<
            CausalMemoryComponentFamily,
            Vec<CausalMemoryDeltaMetrics>,
        > = BTreeMap::new();
        let mut root_groups: BTreeMap<usize, Vec<CausalMemoryDeltaMetrics>> = BTreeMap::new();
        let mut depth_groups: BTreeMap<usize, Vec<CausalMemoryDeltaMetrics>> = BTreeMap::new();
        let mut head_groups: BTreeMap<usize, Vec<CausalMemoryDeltaMetrics>> = BTreeMap::new();
        let mut span_distance_groups: BTreeMap<usize, Vec<CausalMemoryDeltaMetrics>> =
            BTreeMap::new();
        let mut selected_leaf_groups: BTreeMap<usize, Vec<CausalMemoryDeltaMetrics>> =
            BTreeMap::new();
        let mut task_groups: BTreeMap<CausalMemoryTaskFamily, Vec<CausalMemoryDeltaMetrics>> =
            BTreeMap::new();

        for report in &sample_reports {
            for result in &report.interventions {
                let Some(metrics) = result.metrics else {
                    continue;
                };
                intervention_groups
                    .entry(result.intervention)
                    .or_default()
                    .push(metrics);
                if let Some(component_family) = component_family_for(result.intervention) {
                    component_groups
                        .entry(component_family)
                        .or_default()
                        .push(metrics);
                }
                task_groups
                    .entry(report.sample.task_family.clone())
                    .or_default()
                    .push(metrics);
                if let CausalMemoryIntervention::RootDrop { root_index } = result.intervention {
                    root_groups.entry(root_index).or_default().push(metrics);
                }
                for head_context in &result.head_contexts {
                    depth_groups
                        .entry(head_context.routing_depth)
                        .or_default()
                        .push(metrics);
                    head_groups
                        .entry(head_context.head_index)
                        .or_default()
                        .push(metrics);
                    for span_distance in &head_context.span_distances {
                        span_distance_groups
                            .entry(*span_distance)
                            .or_default()
                            .push(metrics);
                    }
                    for selected_leaf_index in &head_context.selected_leaf_indices {
                        selected_leaf_groups
                            .entry(*selected_leaf_index)
                            .or_default()
                            .push(metrics);
                    }
                }
            }
        }

        let utility_by_intervention = aggregate_map(intervention_groups, |intervention, stats| {
            CausalMemoryInterventionAggregate {
                intervention,
                stats,
            }
        });
        let utility_by_component_family =
            aggregate_map(component_groups, |component_family, stats| {
                CausalMemoryComponentFamilyAggregate {
                    component_family,
                    stats,
                }
            });
        let utility_by_root = aggregate_map(root_groups, |root_index, stats| {
            CausalMemoryRootAggregate { root_index, stats }
        });
        let utility_by_routing_depth = aggregate_map(depth_groups, |routing_depth, stats| {
            CausalMemoryRoutingDepthAggregate {
                routing_depth,
                stats,
            }
        });
        let utility_by_routing_head = aggregate_map(head_groups, |head_index, stats| {
            CausalMemoryRoutingHeadAggregate { head_index, stats }
        });
        let utility_by_span_distance =
            aggregate_map(span_distance_groups, |span_distance, stats| {
                CausalMemorySpanDistanceAggregate {
                    span_distance,
                    stats,
                }
            });
        let utility_by_selected_leaf =
            aggregate_map(selected_leaf_groups, |selected_leaf_index, stats| {
                CausalMemorySelectedLeafAggregate {
                    selected_leaf_index,
                    stats,
                }
            });
        let utility_by_task_family = aggregate_map(task_groups, |task_family, stats| {
            CausalMemoryTaskFamilyAggregate { task_family, stats }
        });

        Self {
            tree_retrieval_utility: utility_by_intervention
                .iter()
                .find(|aggregate| aggregate.intervention == CausalMemoryIntervention::NoTreeRead)
                .map(|aggregate| aggregate.stats),
            exact_leaf_read_utility: utility_by_intervention
                .iter()
                .find(|aggregate| {
                    aggregate.intervention == CausalMemoryIntervention::NoExactLeafRead
                })
                .map(|aggregate| aggregate.stats),
            sample_reports,
            utility_by_intervention,
            utility_by_component_family,
            utility_by_root,
            utility_by_routing_depth,
            utility_by_routing_head,
            utility_by_span_distance,
            utility_by_selected_leaf,
            utility_by_task_family,
        }
    }
}

fn component_family_for(
    intervention: CausalMemoryIntervention,
) -> Option<CausalMemoryComponentFamily> {
    match intervention {
        CausalMemoryIntervention::NoTreeRead => {
            Some(CausalMemoryComponentFamily::TreeSummaryRetrieval)
        }
        CausalMemoryIntervention::NoExactLeafRead => {
            Some(CausalMemoryComponentFamily::ExactLeafRead)
        }
        CausalMemoryIntervention::NextBestSpanSubstitution => None,
        CausalMemoryIntervention::RootDrop { .. } => Some(CausalMemoryComponentFamily::LocalTrunk),
    }
}

fn aggregate_map<K, V, F>(groups: BTreeMap<K, Vec<CausalMemoryDeltaMetrics>>, build: F) -> Vec<V>
where
    K: Ord,
    F: Fn(K, CausalMemoryAggregateStats) -> V,
{
    groups
        .into_iter()
        .map(|(key, metrics)| build(key, aggregate_metrics(&metrics)))
        .collect()
}

fn aggregate_metrics(metrics: &[CausalMemoryDeltaMetrics]) -> CausalMemoryAggregateStats {
    let count = metrics.len();
    assert!(count > 0, "aggregate_metrics requires at least one metric");
    let (
        loss_sum,
        target_logit_sum,
        kl_sum,
        retrieval_accuracy_sum,
        retrieval_accuracy_count,
        perplexity_sum,
    ) = metrics.iter().fold(
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0usize, 0.0f32),
        |(
            loss_acc,
            target_acc,
            kl_acc,
            retrieval_accuracy_acc,
            retrieval_accuracy_count_acc,
            perplexity_acc,
        ),
         metric| {
            (
                loss_acc + metric.loss_delta,
                target_acc + metric.target_logit_delta,
                kl_acc + metric.kl_divergence,
                retrieval_accuracy_acc + metric.retrieval_accuracy_delta.unwrap_or(0.0),
                retrieval_accuracy_count_acc
                    + usize::from(metric.retrieval_accuracy_delta.is_some()),
                perplexity_acc + metric.perplexity_delta,
            )
        },
    );

    CausalMemoryAggregateStats {
        count,
        average_loss_delta: loss_sum / count as f32,
        average_target_logit_delta: target_logit_sum / count as f32,
        average_kl_divergence: kl_sum / count as f32,
        retrieval_accuracy_count,
        average_retrieval_accuracy_delta: if retrieval_accuracy_count == 0 {
            None
        } else {
            Some(retrieval_accuracy_sum / retrieval_accuracy_count as f32)
        },
        average_perplexity_delta: perplexity_sum / count as f32,
    }
}

pub(crate) fn summarize_head_contexts(
    head_contexts: &[CausalMemoryHeadContext],
) -> CausalMemoryEvaluationContext {
    let routing_depth = head_contexts
        .iter()
        .map(|head_context| head_context.routing_depth)
        .max()
        .unwrap_or(0);
    let span_distance = head_contexts
        .iter()
        .flat_map(|head_context| head_context.span_distances.iter().copied())
        .min();

    CausalMemoryEvaluationContext {
        routing_depth,
        span_distance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_memory_audit_plan_rejects_duplicate_samples() {
        let error = CausalMemoryAuditPlan::all(vec![
            CausalMemoryAuditSample {
                batch_index: 0,
                position: 7,
                task_family: CausalMemoryTaskFamily::OrdinaryLm,
            },
            CausalMemoryAuditSample {
                batch_index: 0,
                position: 7,
                task_family: CausalMemoryTaskFamily::OrdinaryLm,
            },
        ])
        .unwrap_err();

        assert!(matches!(
            error,
            FractalError::InvalidConfig(message) if message.contains("duplicate sample")
        ));
    }

    #[test]
    fn causal_memory_audit_report_aggregates_by_intervention_root_and_head() {
        let report =
            CausalMemoryAuditReport::from_sample_reports(vec![CausalMemoryAuditSampleReport {
                sample: CausalMemoryAuditSample {
                    batch_index: 0,
                    position: 15,
                    task_family: CausalMemoryTaskFamily::Copy,
                },
                reference_context: CausalMemoryEvaluationContext {
                    routing_depth: 2,
                    span_distance: Some(8),
                },
                reference_head_contexts: vec![CausalMemoryHeadContext {
                    head_index: 0,
                    routing_depth: 2,
                    span_distances: vec![8, 12],
                    selected_leaf_indices: vec![1, 2],
                }],
                target_token_id: 7,
                reference_loss: 1.0,
                reference_target_logit: 0.5,
                interventions: vec![
                    CausalMemoryInterventionResult {
                        intervention: CausalMemoryIntervention::NoTreeRead,
                        applied: true,
                        context: Some(CausalMemoryEvaluationContext {
                            routing_depth: 2,
                            span_distance: Some(8),
                        }),
                        head_contexts: vec![CausalMemoryHeadContext {
                            head_index: 0,
                            routing_depth: 2,
                            span_distances: vec![8, 12],
                            selected_leaf_indices: vec![1, 2],
                        }],
                        metrics: Some(CausalMemoryDeltaMetrics {
                            loss_delta: 0.2,
                            target_logit_delta: 0.3,
                            kl_divergence: 0.4,
                            retrieval_accuracy_delta: Some(0.5),
                            perplexity_delta: 0.5,
                        }),
                    },
                    CausalMemoryInterventionResult {
                        intervention: CausalMemoryIntervention::RootDrop { root_index: 1 },
                        applied: true,
                        context: Some(CausalMemoryEvaluationContext {
                            routing_depth: 2,
                            span_distance: Some(8),
                        }),
                        head_contexts: vec![CausalMemoryHeadContext {
                            head_index: 0,
                            routing_depth: 2,
                            span_distances: vec![8, 12],
                            selected_leaf_indices: vec![1, 2],
                        }],
                        metrics: Some(CausalMemoryDeltaMetrics {
                            loss_delta: 0.6,
                            target_logit_delta: 0.7,
                            kl_divergence: 0.8,
                            retrieval_accuracy_delta: Some(0.9),
                            perplexity_delta: 0.9,
                        }),
                    },
                ],
            }]);

        assert_eq!(
            report.tree_retrieval_utility.unwrap().average_loss_delta,
            0.2
        );
        assert_eq!(report.utility_by_root.len(), 1);
        assert_eq!(report.utility_by_root[0].root_index, 1);
        assert_eq!(report.utility_by_routing_depth.len(), 1);
        assert_eq!(report.utility_by_routing_depth[0].routing_depth, 2);
        assert_eq!(report.utility_by_routing_head.len(), 1);
        assert_eq!(report.utility_by_span_distance.len(), 2);
        assert_eq!(report.utility_by_selected_leaf.len(), 2);
        assert_eq!(report.utility_by_task_family.len(), 1);
        assert_eq!(
            report.utility_by_task_family[0]
                .stats
                .retrieval_accuracy_count,
            2
        );
        assert_eq!(
            report.utility_by_task_family[0]
                .stats
                .average_retrieval_accuracy_delta,
            Some(0.7)
        );
    }

    #[test]
    fn causal_memory_aggregate_stats_expose_retrieval_accuracy_denominator() {
        let report = CausalMemoryAuditReport::from_sample_reports(vec![
            CausalMemoryAuditSampleReport {
                sample: CausalMemoryAuditSample {
                    batch_index: 0,
                    position: 7,
                    task_family: CausalMemoryTaskFamily::OrdinaryLm,
                },
                reference_context: CausalMemoryEvaluationContext {
                    routing_depth: 1,
                    span_distance: Some(2),
                },
                reference_head_contexts: vec![CausalMemoryHeadContext {
                    head_index: 0,
                    routing_depth: 1,
                    span_distances: vec![2],
                    selected_leaf_indices: vec![0],
                }],
                target_token_id: 3,
                reference_loss: 1.0,
                reference_target_logit: 0.5,
                interventions: vec![CausalMemoryInterventionResult {
                    intervention: CausalMemoryIntervention::NoTreeRead,
                    applied: true,
                    context: Some(CausalMemoryEvaluationContext {
                        routing_depth: 1,
                        span_distance: Some(2),
                    }),
                    head_contexts: vec![CausalMemoryHeadContext {
                        head_index: 0,
                        routing_depth: 1,
                        span_distances: vec![2],
                        selected_leaf_indices: vec![0],
                    }],
                    metrics: Some(CausalMemoryDeltaMetrics {
                        loss_delta: 0.1,
                        target_logit_delta: 0.2,
                        kl_divergence: 0.3,
                        retrieval_accuracy_delta: None,
                        perplexity_delta: 0.4,
                    }),
                }],
            },
            CausalMemoryAuditSampleReport {
                sample: CausalMemoryAuditSample {
                    batch_index: 0,
                    position: 15,
                    task_family: CausalMemoryTaskFamily::Copy,
                },
                reference_context: CausalMemoryEvaluationContext {
                    routing_depth: 1,
                    span_distance: Some(4),
                },
                reference_head_contexts: vec![CausalMemoryHeadContext {
                    head_index: 0,
                    routing_depth: 1,
                    span_distances: vec![4],
                    selected_leaf_indices: vec![1],
                }],
                target_token_id: 5,
                reference_loss: 1.2,
                reference_target_logit: 0.7,
                interventions: vec![CausalMemoryInterventionResult {
                    intervention: CausalMemoryIntervention::NoTreeRead,
                    applied: true,
                    context: Some(CausalMemoryEvaluationContext {
                        routing_depth: 1,
                        span_distance: Some(4),
                    }),
                    head_contexts: vec![CausalMemoryHeadContext {
                        head_index: 0,
                        routing_depth: 1,
                        span_distances: vec![4],
                        selected_leaf_indices: vec![1],
                    }],
                    metrics: Some(CausalMemoryDeltaMetrics {
                        loss_delta: 0.5,
                        target_logit_delta: 0.6,
                        kl_divergence: 0.7,
                        retrieval_accuracy_delta: Some(1.0),
                        perplexity_delta: 0.8,
                    }),
                }],
            },
        ]);

        let stats = report
            .utility_by_intervention
            .iter()
            .find(|aggregate| aggregate.intervention == CausalMemoryIntervention::NoTreeRead)
            .unwrap()
            .stats;

        assert_eq!(stats.count, 2);
        assert_eq!(stats.retrieval_accuracy_count, 1);
        assert_eq!(stats.average_retrieval_accuracy_delta, Some(1.0));
    }
}
