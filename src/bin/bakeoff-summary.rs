use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use fractal::{
    lifecycle::BenchmarkMode, species_registry_for_species, ComparisonAuthority,
    ComparisonContract, SpeciesId, SpeciesRawMetrics,
};
use fractal_eval_private::aggregate_results;
use serde::Deserialize;

const DEFAULT_ROOT: &str = ".runpod-local-logs/runpod-results";

fn main() {
    if let Err(error) = run() {
        eprintln!("bakeoff-summary: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = CliArgs::parse(std::env::args().skip(1))?;
    let output = summarize_root(&args.root, args.report)?;
    print!("{output}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    root: PathBuf,
    report: ReportKind,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut root = PathBuf::from(DEFAULT_ROOT);
        let mut report = ReportKind::Leaderboard;
        let mut saw_help = false;
        let mut iter = args.peekable();

        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--root" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--root requires a path argument".to_owned())?;
                    root = PathBuf::from(value);
                }
                "--report" => {
                    let value = iter
                        .next()
                        .ok_or_else(|| "--report requires a value".to_owned())?;
                    report = ReportKind::parse(&value)?;
                }
                "--help" | "-h" => {
                    saw_help = true;
                }
                _ => return Err(format!("unknown argument: {arg}")),
            }
        }

        if saw_help {
            println!("{}", usage());
            std::process::exit(0);
        }

        Ok(Self { root, report })
    }
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin bakeoff-summary -- [--root <path>] [--report <leaderboard|systems-speed|ledger>]"
    );
    let _ = writeln!(output, "Default root: {DEFAULT_ROOT}");
    output
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReportKind {
    Leaderboard,
    SystemsSpeed,
    Ledger,
}

impl ReportKind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "leaderboard" => Ok(Self::Leaderboard),
            "systems-speed" | "systems_speed" => Ok(Self::SystemsSpeed),
            "ledger" => Ok(Self::Ledger),
            _ => Err(format!("unknown report kind: {value}")),
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Leaderboard => "Bakeoff Summary",
            Self::SystemsSpeed => "Systems-Speed Summary",
            Self::Ledger => "Experiment Ledger",
        }
    }
}

#[derive(Clone, Debug)]
struct LoadedRows {
    scanned_root: PathBuf,
    completed: Vec<CompletedRow>,
    failures: Vec<FailureRow>,
    pending: Vec<PendingRow>,
}

fn summarize_root(root: &Path, report: ReportKind) -> Result<String, String> {
    let loaded = load_rows(root)?;
    match report {
        ReportKind::Leaderboard => summarize_leaderboard(&loaded),
        ReportKind::SystemsSpeed => summarize_systems_speed(&loaded),
        ReportKind::Ledger => summarize_ledger(&loaded),
    }
}

fn load_rows(root: &Path) -> Result<LoadedRows, String> {
    let mut scanned_root = root.to_path_buf();
    let mut run_dirs = discover_run_dirs(&scanned_root)?;
    if run_dirs.is_empty() && root.ends_with("runpod-results") {
        let fallback = root
            .parent()
            .ok_or_else(|| "unable to determine fallback root".to_owned())?;
        run_dirs = discover_run_dirs(fallback)?;
        scanned_root = fallback.to_path_buf();
    }

    let mut runs = Vec::new();
    for run_dir in run_dirs {
        runs.push(load_run(&run_dir)?);
    }

    Ok(LoadedRows {
        scanned_root,
        completed: dedupe_completed_rows(
            runs.iter()
                .flat_map(|run| run.completed_rows.iter().cloned())
                .collect(),
        ),
        failures: runs
            .iter()
            .flat_map(|run| run.failed_rows.iter().cloned())
            .collect(),
        pending: runs
            .iter()
            .flat_map(|run| run.pending_rows.iter().cloned())
            .collect(),
    })
}

fn summarize_leaderboard(loaded: &LoadedRows) -> Result<String, String> {
    let completed: Vec<_> = loaded
        .completed
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::Leaderboard)
        .cloned()
        .collect();
    let failures: Vec<_> = loaded
        .failures
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::Leaderboard)
        .cloned()
        .collect();
    let pending: Vec<_> = loaded
        .pending
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::Leaderboard)
        .cloned()
        .collect();

    let mut output = String::new();
    writeln!(output, "{}", ReportKind::Leaderboard.title()).unwrap();
    writeln!(output, "root: {}", loaded.scanned_root.display()).unwrap();
    writeln!(output, "completed rows: {}", completed.len()).unwrap();
    writeln!(output, "failure rows: {}", failures.len()).unwrap();
    writeln!(output, "pending runs: {}", pending.len()).unwrap();

    let grouped = group_completed_by_seed_and_contract(&completed);
    writeln!(output).unwrap();
    writeln!(output, "Per-seed leaderboards:").unwrap();
    if grouped.is_empty() {
        writeln!(output, "(none yet)").unwrap();
    } else {
        for (group_key, mut rows) in grouped {
            rows.sort_by(|left, right| {
                right
                    .fitness
                    .partial_cmp(&left.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.variant_name.cmp(&right.variant_name))
            });

            writeln!(
                output,
                "seed={} | preset={} | authority={} | runtime={}{}",
                group_key.seed_label(),
                group_key.preset,
                group_key.comparison_label(),
                group_key.runtime_surface_policy,
                group_key
                    .lane
                    .as_deref()
                    .map(|lane| format!(" | lane={lane}"))
                    .unwrap_or_default()
            )
            .unwrap();
            output.push_str(&render_completed_table(&rows));
            writeln!(output).unwrap();

            let group_failures: Vec<_> = failures
                .iter()
                .filter(|failure| failure.matches_seed_and_contract(&group_key))
                .cloned()
                .collect();
            if !group_failures.is_empty() {
                writeln!(output, "failures").unwrap();
                output.push_str(&render_failure_table(&group_failures));
                writeln!(output).unwrap();
            }
        }
    }

    let authoritative_aggregates = aggregate_rows(
        completed
            .iter()
            .filter(|row| row.comparison.authority == ComparisonAuthority::Authoritative)
            .cloned()
            .collect(),
    );
    let advisory_aggregates = aggregate_rows(
        completed
            .iter()
            .filter(|row| row.comparison.authority == ComparisonAuthority::Advisory)
            .cloned()
            .collect(),
    );

    writeln!(output, "Aggregate authoritative leaderboard:").unwrap();
    if authoritative_aggregates.is_empty() {
        writeln!(output, "(none yet)").unwrap();
    } else {
        for ((preset, runtime_surface_policy), rows) in authoritative_aggregates {
            writeln!(output, "preset={preset} | runtime={runtime_surface_policy}").unwrap();
            output.push_str(&render_aggregate_table(&rows));
            writeln!(output).unwrap();
        }
    }

    if !advisory_aggregates.is_empty() {
        writeln!(output, "Advisory snapshot:").unwrap();
        for ((preset, runtime_surface_policy), rows) in advisory_aggregates {
            writeln!(output, "preset={preset} | runtime={runtime_surface_policy}").unwrap();
            output.push_str(&render_aggregate_table(&rows));
            writeln!(output).unwrap();
        }
    }

    if !pending.is_empty() {
        writeln!(output, "Pending runs:").unwrap();
        output.push_str(&render_pending_table(&pending));
    }

    Ok(output)
}

fn summarize_systems_speed(loaded: &LoadedRows) -> Result<String, String> {
    let completed: Vec<_> = loaded
        .completed
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::SystemsSpeed)
        .cloned()
        .collect();
    let failures: Vec<_> = loaded
        .failures
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::SystemsSpeed)
        .cloned()
        .collect();
    let pending: Vec<_> = loaded
        .pending
        .iter()
        .filter(|row| row.benchmark_mode == BenchmarkMode::SystemsSpeed)
        .cloned()
        .collect();

    let mut output = String::new();
    writeln!(output, "{}", ReportKind::SystemsSpeed.title()).unwrap();
    writeln!(output, "root: {}", loaded.scanned_root.display()).unwrap();
    writeln!(output, "completed rows: {}", completed.len()).unwrap();
    writeln!(output, "failure rows: {}", failures.len()).unwrap();
    writeln!(output, "pending runs: {}", pending.len()).unwrap();

    let aggregates = aggregate_rows(completed.clone());
    writeln!(output).unwrap();
    writeln!(output, "Aggregate systems-speed leaderboard:").unwrap();
    if aggregates.is_empty() {
        writeln!(output, "(none yet)").unwrap();
    } else {
        for ((preset, runtime_surface_policy), mut rows) in aggregates {
            rows.sort_by(|left, right| {
                right
                    .tok_s
                    .partial_cmp(&left.tok_s)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        right
                            .fitness
                            .partial_cmp(&left.fitness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
            writeln!(output, "preset={preset} | runtime={runtime_surface_policy}").unwrap();
            output.push_str(&render_aggregate_table(&rows));
            writeln!(output).unwrap();
        }
    }

    if !failures.is_empty() {
        writeln!(output, "systems-speed failures").unwrap();
        output.push_str(&render_failure_table(&failures));
        writeln!(output).unwrap();
    }
    if !pending.is_empty() {
        writeln!(output, "systems-speed pending").unwrap();
        output.push_str(&render_pending_table(&pending));
    }

    Ok(output)
}

fn summarize_ledger(loaded: &LoadedRows) -> Result<String, String> {
    let ledger = build_ledger_entries(loaded);
    let mut output = String::new();
    writeln!(output, "{}", ReportKind::Ledger.title()).unwrap();
    writeln!(output, "root: {}", loaded.scanned_root.display()).unwrap();
    writeln!(output, "entries: {}", ledger.len()).unwrap();
    writeln!(output).unwrap();
    if ledger.is_empty() {
        writeln!(output, "(none yet)").unwrap();
    } else {
        output.push_str(&render_ledger_table(&ledger));
    }
    Ok(output)
}

fn discover_run_dirs(root: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let mut run_dirs = BTreeSet::new();
    if !root.exists() {
        return Ok(run_dirs);
    }
    discover_run_dirs_recursive(root, &mut run_dirs)?;
    Ok(run_dirs)
}

fn discover_run_dirs_recursive(dir: &Path, run_dirs: &mut BTreeSet<PathBuf>) -> Result<(), String> {
    if is_run_dir(dir) {
        run_dirs.insert(dir.to_path_buf());
    }

    for entry in fs::read_dir(dir).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        if entry.file_type().map_err(io_error)?.is_dir() {
            discover_run_dirs_recursive(&entry.path(), run_dirs)?;
        }
    }
    Ok(())
}

fn is_run_dir(dir: &Path) -> bool {
    dir.join("metadata/wrapper-manifest.json").exists()
        || dir.join("remote/manifests/run-manifest.json").exists()
        || dir
            .join("remote/artifacts/tournament-run-artifact.json")
            .exists()
        || dir.join("manifests/tournament-run-manifest.json").exists()
        || dir.join("artifacts/tournament-run-artifact.json").exists()
}

fn load_run(run_dir: &Path) -> Result<RunSummary, String> {
    let wrapper_manifest =
        read_json::<RunControlManifest>(&run_dir.join("metadata/wrapper-manifest.json"))?;
    let remote_manifest =
        read_json::<RunControlManifest>(&run_dir.join("remote/manifests/run-manifest.json"))?
            .or_else(|| {
                read_json::<RunControlManifest>(
                    &run_dir.join("manifests/tournament-run-manifest.json"),
                )
                .ok()
                .flatten()
            });
    let artifact = read_json::<RunArtifactFile>(
        &run_dir.join("remote/artifacts/tournament-run-artifact.json"),
    )?
    .or_else(|| {
        read_json::<RunArtifactFile>(&run_dir.join("artifacts/tournament-run-artifact.json"))
            .ok()
            .flatten()
    });

    let metadata = RunMetadata::from_sources(
        run_dir,
        wrapper_manifest.as_ref(),
        remote_manifest.as_ref(),
        artifact.as_ref(),
    );
    let mut completed_rows = Vec::new();
    let mut failed_rows = Vec::new();
    let mut pending_rows = Vec::new();

    if let Some(artifact) = artifact.as_ref() {
        for record in &artifact.results {
            if record.outcome_class.as_deref() == Some("success") {
                if let Some(row) =
                    completed_row_from_artifact(&metadata, record, artifact.manifest.as_ref())
                {
                    completed_rows.push(row);
                } else {
                    failed_rows.push(FailureRow {
                        run_id: metadata.run_id.clone(),
                        seed: metadata.seed,
                        preset: metadata.preset.clone(),
                        lane: metadata.lane.clone(),
                        comparison: metadata.comparison.clone(),
                        runtime_surface_policy: metadata.runtime_surface_policy.clone(),
                        benchmark_mode: metadata.benchmark_mode,
                        logical_name: metadata.logical_name.clone(),
                        question_summary: metadata.question_summary.clone(),
                        commit_sha: metadata.commit_sha.clone(),
                        created_at_unix_ms: metadata.created_at_unix_ms,
                        backend: metadata.backend.clone(),
                        variant_name: record
                            .variant_name
                            .clone()
                            .or_else(|| record.species.clone()),
                        species: record
                            .species
                            .as_deref()
                            .and_then(|value| value.parse().ok()),
                        outcome: "numeric-failure".to_owned(),
                        error: "success artifact lacked metrics".to_owned(),
                    });
                }
            } else {
                failed_rows.push(FailureRow {
                    run_id: metadata.run_id.clone(),
                    seed: metadata.seed,
                    preset: metadata.preset.clone(),
                    lane: metadata.lane.clone(),
                    comparison: record
                        .comparison_contract()
                        .or_else(|| metadata.comparison.clone()),
                    runtime_surface_policy: record
                        .runtime_surface_policy
                        .clone()
                        .or_else(|| metadata.runtime_surface_policy.clone()),
                    benchmark_mode: record
                        .experiment
                        .as_ref()
                        .and_then(ArtifactExperimentRecord::benchmark_mode)
                        .unwrap_or(metadata.benchmark_mode),
                    logical_name: record
                        .experiment
                        .as_ref()
                        .and_then(|experiment| experiment.experiment_id.as_ref())
                        .and_then(|id| id.logical_name.clone())
                        .or_else(|| metadata.logical_name.clone()),
                    question_summary: record
                        .experiment
                        .as_ref()
                        .and_then(|experiment| experiment.question.as_ref())
                        .and_then(|question| question.summary.clone())
                        .or_else(|| metadata.question_summary.clone()),
                    commit_sha: record
                        .experiment
                        .as_ref()
                        .and_then(|experiment| experiment.experiment_id.as_ref())
                        .and_then(|id| id.commit_sha.clone())
                        .or_else(|| metadata.commit_sha.clone()),
                    created_at_unix_ms: record
                        .experiment
                        .as_ref()
                        .and_then(|experiment| experiment.experiment_id.as_ref())
                        .and_then(|id| id.created_at_unix_ms)
                        .or(metadata.created_at_unix_ms),
                    backend: record
                        .experiment
                        .as_ref()
                        .and_then(|experiment| experiment.execution.as_ref())
                        .and_then(|execution| execution.backend.clone())
                        .or_else(|| metadata.backend.clone()),
                    variant_name: record
                        .variant_name
                        .clone()
                        .or_else(|| record.species.clone()),
                    species: record
                        .species
                        .as_deref()
                        .and_then(|value| value.parse().ok()),
                    outcome: record
                        .outcome_class
                        .clone()
                        .unwrap_or_else(|| "unknown".to_owned()),
                    error: record
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown error".to_owned()),
                });
            }
        }
    } else if let Some(manifest) = remote_manifest.as_ref() {
        if manifest.status.as_deref() == Some("failure") || manifest.exit_code.unwrap_or(0) != 0 {
            failed_rows.push(FailureRow {
                run_id: metadata.run_id.clone(),
                seed: metadata.seed,
                preset: metadata.preset.clone(),
                lane: metadata.lane.clone(),
                comparison: metadata.comparison.clone(),
                runtime_surface_policy: metadata.runtime_surface_policy.clone(),
                benchmark_mode: metadata.benchmark_mode,
                logical_name: metadata.logical_name.clone(),
                question_summary: metadata.question_summary.clone(),
                commit_sha: metadata.commit_sha.clone(),
                created_at_unix_ms: metadata.created_at_unix_ms,
                backend: metadata.backend.clone(),
                variant_name: metadata.variant_name.clone(),
                species: metadata.species,
                outcome: "infra-failure".to_owned(),
                error: format!(
                    "run failed before artifact capture (exit_code={})",
                    manifest.exit_code.unwrap_or(-1)
                ),
            });
        } else {
            pending_rows.push(PendingRow {
                run_id: metadata.run_id.clone(),
                seed: metadata.seed,
                preset: metadata.preset.clone(),
                lane: metadata.lane.clone(),
                comparison: metadata.comparison.clone(),
                runtime_surface_policy: metadata.runtime_surface_policy.clone(),
                benchmark_mode: metadata.benchmark_mode,
                logical_name: metadata.logical_name.clone(),
                question_summary: metadata.question_summary.clone(),
                commit_sha: metadata.commit_sha.clone(),
                created_at_unix_ms: metadata.created_at_unix_ms,
                backend: metadata.backend.clone(),
                variant_name: metadata.variant_name.clone(),
                species: metadata.species,
                status: manifest
                    .status
                    .clone()
                    .unwrap_or_else(|| "pending".to_owned()),
                pod_id: metadata.pod_id.clone(),
            });
        }
    } else {
        pending_rows.push(PendingRow {
            run_id: metadata.run_id.clone(),
            seed: metadata.seed,
            preset: metadata.preset.clone(),
            lane: metadata.lane.clone(),
            comparison: metadata.comparison.clone(),
            runtime_surface_policy: metadata.runtime_surface_policy.clone(),
            benchmark_mode: metadata.benchmark_mode,
            logical_name: metadata.logical_name.clone(),
            question_summary: metadata.question_summary.clone(),
            commit_sha: metadata.commit_sha.clone(),
            created_at_unix_ms: metadata.created_at_unix_ms,
            backend: metadata.backend.clone(),
            variant_name: metadata.variant_name.clone(),
            species: metadata.species,
            status: wrapper_manifest
                .as_ref()
                .and_then(|manifest| manifest.status.clone())
                .unwrap_or_else(|| "pending".to_owned()),
            pod_id: metadata.pod_id.clone(),
        });
    }

    Ok(RunSummary {
        metadata,
        completed_rows,
        failed_rows,
        pending_rows,
    })
}

fn completed_row_from_artifact(
    metadata: &RunMetadata,
    record: &ArtifactSpeciesRecord,
    manifest: Option<&RunManifestFile>,
) -> Option<CompletedRow> {
    let species = record
        .species
        .as_deref()
        .and_then(|value| value.parse::<SpeciesId>().ok())
        .or(metadata.species)?;
    let variant_name = record
        .variant_name
        .clone()
        .unwrap_or_else(|| species_variant_name(species));
    let result = if let Some(ranked) = record.ranked_result.as_ref() {
        RankedResult {
            rank: ranked.rank.unwrap_or(0),
            fitness: ranked.fitness?,
            stability: ranked.stability_score?,
            perplexity: ranked.long_context_perplexity?,
            arc: ranked.arc_accuracy?,
            tok_s: ranked.tokens_per_sec?,
        }
    } else if let Some(metrics) = record.metrics.as_ref() {
        let metrics = metrics.to_species_raw_metrics(species)?;
        let ranked = aggregate_results(vec![metrics]).into_iter().next()?;
        RankedResult {
            rank: ranked.rank,
            fitness: ranked.fitness,
            stability: ranked.stability_score,
            perplexity: ranked.long_context_perplexity,
            arc: ranked.arc_accuracy,
            tok_s: ranked.tokens_per_sec,
        }
    } else {
        return None;
    };

    Some(CompletedRow {
        run_id: metadata.run_id.clone(),
        seed: metadata.seed,
        preset: metadata
            .preset
            .clone()
            .or_else(|| manifest.and_then(|manifest| manifest.preset.clone()))
            .unwrap_or_else(|| "unknown".to_owned()),
        lane: metadata.lane.clone(),
        comparison: record
            .comparison_contract()
            .or_else(|| metadata.comparison.clone())
            .or_else(|| manifest.and_then(|manifest| manifest.comparison_contract()))
            .unwrap_or_else(ComparisonContract::authoritative_same_preset),
        runtime_surface_policy: record
            .runtime_surface_policy
            .clone()
            .or_else(|| manifest.and_then(|manifest| manifest.runtime_surface_policy.clone()))
            .or_else(|| metadata.runtime_surface_policy.clone())
            .unwrap_or_else(|| "conservative-defaults".to_owned()),
        benchmark_mode: record
            .experiment
            .as_ref()
            .and_then(ArtifactExperimentRecord::benchmark_mode)
            .unwrap_or(metadata.benchmark_mode),
        logical_name: record
            .experiment
            .as_ref()
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.logical_name.clone())
            .or_else(|| metadata.logical_name.clone()),
        question_summary: record
            .experiment
            .as_ref()
            .and_then(|experiment| experiment.question.as_ref())
            .and_then(|question| question.summary.clone())
            .or_else(|| metadata.question_summary.clone()),
        commit_sha: record
            .experiment
            .as_ref()
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.commit_sha.clone())
            .or_else(|| metadata.commit_sha.clone()),
        created_at_unix_ms: record
            .experiment
            .as_ref()
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.created_at_unix_ms)
            .or(metadata.created_at_unix_ms),
        backend: record
            .experiment
            .as_ref()
            .and_then(|experiment| experiment.execution.as_ref())
            .and_then(|execution| execution.backend.clone())
            .or_else(|| metadata.backend.clone()),
        variant_name,
        species,
        rank: result.rank,
        fitness: result.fitness,
        stability: result.stability,
        perplexity: result.perplexity,
        arc: result.arc,
        tok_s: result.tok_s,
    })
}

fn species_variant_name(species: SpeciesId) -> String {
    species_registry_for_species(species)
        .first()
        .map(|definition| definition.variant_name.as_str().to_owned())
        .unwrap_or_else(|| species.as_str().to_owned())
}

type CompletedRowDedupKey = (Option<u64>, String, String, String, String, String);

fn dedupe_completed_rows(mut rows: Vec<CompletedRow>) -> Vec<CompletedRow> {
    let mut best_by_key: BTreeMap<CompletedRowDedupKey, CompletedRow> = BTreeMap::new();
    for row in rows.drain(..) {
        let key = (
            row.seed,
            row.preset.clone(),
            row.variant_name.clone(),
            row.comparison.label().to_owned(),
            row.runtime_surface_policy.clone(),
            row.benchmark_mode.as_str().to_owned(),
        );
        best_by_key
            .entry(key)
            .and_modify(|current| {
                if row.run_id > current.run_id {
                    *current = row.clone();
                }
            })
            .or_insert(row);
    }
    best_by_key.into_values().collect()
}

fn group_completed_by_seed_and_contract(
    rows: &[CompletedRow],
) -> BTreeMap<RunGroupKey, Vec<CompletedRow>> {
    let mut groups: BTreeMap<RunGroupKey, Vec<CompletedRow>> = BTreeMap::new();
    for row in rows {
        groups
            .entry(RunGroupKey::from_row(row))
            .or_default()
            .push(row.clone());
    }
    groups
}

fn aggregate_rows(rows: Vec<CompletedRow>) -> BTreeMap<(String, String), Vec<AggregateRow>> {
    let mut grouped: BTreeMap<(String, String, String, String), Vec<CompletedRow>> =
        BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.preset.clone(),
                row.comparison.label().to_owned(),
                row.runtime_surface_policy.clone(),
                row.variant_name.clone(),
            ))
            .or_default()
            .push(row);
    }

    let mut output: BTreeMap<(String, String), Vec<AggregateRow>> = BTreeMap::new();
    for ((preset, comparison_label, runtime_surface_policy, variant_name), entries) in grouped {
        let count = entries.len() as f64;
        let mut rank = 0usize;
        let mut fitness = 0.0;
        let mut stability = 0.0;
        let mut perplexity = 0.0;
        let mut arc = 0.0;
        let mut tok_s = 0.0;

        for entry in &entries {
            rank = rank.max(entry.rank);
            fitness += entry.fitness;
            stability += entry.stability;
            perplexity += entry.perplexity;
            arc += entry.arc;
            tok_s += entry.tok_s;
        }

        output
            .entry((preset, runtime_surface_policy))
            .or_default()
            .push(AggregateRow {
                variant_name,
                samples: entries.len(),
                rank,
                comparison: parse_comparison_contract(None, Some(&comparison_label))
                    .unwrap_or_else(ComparisonContract::authoritative_same_preset),
                fitness: fitness / count,
                stability: stability / count,
                perplexity: perplexity / count,
                arc: arc / count,
                tok_s: tok_s / count,
            });
    }

    for rows in output.values_mut() {
        rows.sort_by(|left, right| {
            right
                .fitness
                .partial_cmp(&left.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.variant_name.cmp(&right.variant_name))
        });
    }

    output
}

fn render_completed_table(rows: &[CompletedRow]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            vec![
                row.rank.to_string(),
                row.variant_name.clone(),
                format!("{:.2}", row.fitness),
                format!("{:.2}", row.stability),
                format!("{:.2}", row.perplexity),
                format!("{:.2}", row.arc),
                format!("{:.0}", row.tok_s),
                row.run_id.clone(),
            ]
        })
        .collect::<Vec<_>>();
    render_table(
        &[
            "rank",
            "variant",
            "fitness",
            "stability",
            "perplexity",
            "ARC",
            "tok/s",
            "run_id",
        ],
        &rows,
    )
}

fn render_aggregate_table(rows: &[AggregateRow]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            vec![
                row.variant_name.clone(),
                row.samples.to_string(),
                row.comparison.label().to_owned(),
                format!("{:.2}", row.fitness),
                format!("{:.2}", row.stability),
                format!("{:.2}", row.perplexity),
                format!("{:.2}", row.arc),
                format!("{:.0}", row.tok_s),
            ]
        })
        .collect::<Vec<_>>();
    render_table(
        &[
            "variant",
            "samples",
            "authority",
            "fitness",
            "stability",
            "perplexity",
            "ARC",
            "tok/s",
        ],
        &rows,
    )
}

fn render_failure_table(rows: &[FailureRow]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            vec![
                row.variant_name
                    .clone()
                    .unwrap_or_else(|| "(unknown)".to_owned()),
                row.outcome.clone(),
                row.error.clone(),
                row.run_id.clone(),
            ]
        })
        .collect::<Vec<_>>();
    render_table(&["variant", "outcome", "error", "run_id"], &rows)
}

fn render_pending_table(rows: &[PendingRow]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            vec![
                row.variant_name
                    .clone()
                    .unwrap_or_else(|| "(unknown)".to_owned()),
                row.status.clone(),
                row.preset.clone().unwrap_or_else(|| "(unknown)".to_owned()),
                row.seed
                    .map(|seed| seed.to_string())
                    .unwrap_or_else(|| "?".to_owned()),
                row.pod_id.clone().unwrap_or_else(|| "(unknown)".to_owned()),
                row.run_id.clone(),
            ]
        })
        .collect::<Vec<_>>();
    render_table(
        &["variant", "status", "preset", "seed", "pod_id", "run_id"],
        &rows,
    )
}

fn render_ledger_table(rows: &[LedgerEntry]) -> String {
    let rows = rows
        .iter()
        .map(|row| {
            vec![
                row.when.clone(),
                row.change_axis.clone(),
                row.variant_name.clone(),
                row.preset.clone(),
                row.benchmark_mode.as_str().to_owned(),
                row.outcome.clone(),
                row.fitness
                    .map(|value| format!("{value:.2}"))
                    .unwrap_or_else(|| "-".to_owned()),
                row.tok_s
                    .map(|value| format!("{value:.0}"))
                    .unwrap_or_else(|| "-".to_owned()),
                row.commit_sha.clone().unwrap_or_else(|| "-".to_owned()),
                row.logical_name.clone().unwrap_or_else(|| "-".to_owned()),
                row.question_summary
                    .clone()
                    .unwrap_or_else(|| "-".to_owned()),
            ]
        })
        .collect::<Vec<_>>();
    render_table(
        &[
            "when",
            "change",
            "variant",
            "preset",
            "benchmark_mode",
            "outcome",
            "fitness",
            "tok/s",
            "commit",
            "logical_name",
            "question",
        ],
        &rows,
    )
}

fn build_ledger_entries(loaded: &LoadedRows) -> Vec<LedgerEntry> {
    let mut entries = Vec::new();
    for row in &loaded.completed {
        entries.push(LedgerEntry {
            when_unix_ms: row.created_at_unix_ms.unwrap_or(0),
            when: ledger_when(row.created_at_unix_ms, &row.run_id),
            run_id: row.run_id.clone(),
            logical_name: row.logical_name.clone(),
            question_summary: row.question_summary.clone(),
            variant_name: row.variant_name.clone(),
            preset: row.preset.clone(),
            benchmark_mode: row.benchmark_mode,
            runtime_surface_policy: row.runtime_surface_policy.clone(),
            commit_sha: row.commit_sha.clone(),
            outcome: "success".to_owned(),
            fitness: Some(row.fitness),
            tok_s: Some(row.tok_s),
            change_axis: String::new(),
        });
    }
    for row in &loaded.failures {
        entries.push(LedgerEntry {
            when_unix_ms: row.created_at_unix_ms.unwrap_or(0),
            when: ledger_when(row.created_at_unix_ms, &row.run_id),
            run_id: row.run_id.clone(),
            logical_name: row.logical_name.clone(),
            question_summary: row.question_summary.clone(),
            variant_name: row
                .variant_name
                .clone()
                .unwrap_or_else(|| "(unknown)".to_owned()),
            preset: row.preset.clone().unwrap_or_else(|| "(unknown)".to_owned()),
            benchmark_mode: row.benchmark_mode,
            runtime_surface_policy: row
                .runtime_surface_policy
                .clone()
                .unwrap_or_else(|| "(unknown)".to_owned()),
            commit_sha: row.commit_sha.clone(),
            outcome: row.outcome.clone(),
            fitness: None,
            tok_s: None,
            change_axis: String::new(),
        });
    }
    for row in &loaded.pending {
        entries.push(LedgerEntry {
            when_unix_ms: row.created_at_unix_ms.unwrap_or(0),
            when: ledger_when(row.created_at_unix_ms, &row.run_id),
            run_id: row.run_id.clone(),
            logical_name: row.logical_name.clone(),
            question_summary: row.question_summary.clone(),
            variant_name: row
                .variant_name
                .clone()
                .unwrap_or_else(|| "(unknown)".to_owned()),
            preset: row.preset.clone().unwrap_or_else(|| "(unknown)".to_owned()),
            benchmark_mode: row.benchmark_mode,
            runtime_surface_policy: row
                .runtime_surface_policy
                .clone()
                .unwrap_or_else(|| "(unknown)".to_owned()),
            commit_sha: row.commit_sha.clone(),
            outcome: format!("pending:{}", row.status),
            fitness: None,
            tok_s: None,
            change_axis: String::new(),
        });
    }

    entries.sort_by(|left, right| {
        left.when_unix_ms
            .cmp(&right.when_unix_ms)
            .then_with(|| left.run_id.cmp(&right.run_id))
    });

    let mut previous_by_stream: BTreeMap<(String, String, String), LedgerEntry> = BTreeMap::new();
    for entry in &mut entries {
        let stream_key = (
            entry.benchmark_mode.as_str().to_owned(),
            entry.preset.clone(),
            entry.runtime_surface_policy.clone(),
        );
        entry.change_axis =
            classify_change_axis(entry, previous_by_stream.get(&stream_key)).to_owned();
        previous_by_stream.insert(stream_key, entry.clone());
    }

    entries
}

fn classify_change_axis(current: &LedgerEntry, previous: Option<&LedgerEntry>) -> &'static str {
    let Some(previous) = previous else {
        return "new";
    };
    if current.outcome == "infra-failure" {
        return "infra";
    }
    if current.variant_name != previous.variant_name {
        return "primitive";
    }
    if current.benchmark_mode != previous.benchmark_mode
        || current.runtime_surface_policy != previous.runtime_surface_policy
    {
        return "runtime";
    }
    if current.preset != previous.preset {
        return "budget";
    }
    if current.commit_sha != previous.commit_sha {
        return "infra";
    }
    "repeat"
}

fn ledger_when(timestamp: Option<u64>, run_id: &str) -> String {
    timestamp
        .map(|value| value.to_string())
        .unwrap_or_else(|| run_id.to_owned())
}

fn render_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut widths = headers
        .iter()
        .map(|header| header.len())
        .collect::<Vec<_>>();
    for row in rows {
        for (index, cell) in row.iter().enumerate() {
            if index >= widths.len() {
                widths.push(cell.len());
            } else {
                widths[index] = widths[index].max(cell.len());
            }
        }
    }

    let mut output = String::new();
    let header = headers
        .iter()
        .enumerate()
        .map(|(index, header)| pad_cell(header, widths[index]))
        .collect::<Vec<_>>()
        .join("  ");
    let separator = widths
        .iter()
        .map(|width| "-".repeat(*width))
        .collect::<Vec<_>>()
        .join("  ");
    writeln!(output, "{header}").unwrap();
    writeln!(output, "{separator}").unwrap();
    for row in rows {
        let line = row
            .iter()
            .enumerate()
            .map(|(index, cell)| pad_cell(cell, widths[index]))
            .collect::<Vec<_>>()
            .join("  ");
        writeln!(output, "{line}").unwrap();
    }
    output
}

fn pad_cell(value: &str, width: usize) -> String {
    format!("{value:<width$}", width = width)
}

fn parse_benchmark_mode_label(value: &str) -> Option<BenchmarkMode> {
    match value {
        "leaderboard" => Some(BenchmarkMode::Leaderboard),
        "systems-speed" | "systems_speed" => Some(BenchmarkMode::SystemsSpeed),
        _ => None,
    }
}

fn read_json<T>(path: &Path) -> Result<Option<T>, String>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(None);
    }
    let file = fs::File::open(path).map_err(io_error)?;
    let value = serde_json::from_reader(file)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    Ok(Some(value))
}

fn io_error(error: std::io::Error) -> String {
    error.to_string()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum SeedLabel {
    Known(u64),
    Unknown,
}

impl SeedLabel {
    fn from_option(seed: Option<u64>) -> Self {
        seed.map(SeedLabel::Known).unwrap_or(SeedLabel::Unknown)
    }

    fn label(self) -> String {
        match self {
            Self::Known(seed) => seed.to_string(),
            Self::Unknown => "?".to_owned(),
        }
    }

    fn as_option(self) -> Option<u64> {
        match self {
            Self::Known(seed) => Some(seed),
            Self::Unknown => None,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct RunSummary {
    #[allow(dead_code)]
    metadata: RunMetadata,
    completed_rows: Vec<CompletedRow>,
    failed_rows: Vec<FailureRow>,
    pending_rows: Vec<PendingRow>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct RunMetadata {
    run_id: String,
    seed: Option<u64>,
    preset: Option<String>,
    lane: Option<String>,
    comparison: Option<ComparisonContract>,
    runtime_surface_policy: Option<String>,
    benchmark_mode: BenchmarkMode,
    logical_name: Option<String>,
    question_summary: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
    backend: Option<String>,
    pod_id: Option<String>,
    species: Option<SpeciesId>,
    variant_name: Option<String>,
}

impl RunMetadata {
    fn from_sources(
        run_dir: &Path,
        wrapper_manifest: Option<&RunControlManifest>,
        remote_manifest: Option<&RunControlManifest>,
        artifact: Option<&RunArtifactFile>,
    ) -> Self {
        let explicit_experiment = wrapper_manifest
            .and_then(|manifest| manifest.experiment.as_ref())
            .or_else(|| remote_manifest.and_then(|manifest| manifest.experiment.as_ref()));
        let artifact_experiment = artifact
            .and_then(|artifact| artifact.results.first())
            .and_then(|record| record.experiment.as_ref())
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.experiments.first())
            });

        let seed = explicit_experiment
            .and_then(|experiment| experiment.resolved.as_ref())
            .and_then(|resolved| resolved.seed)
            .or_else(|| wrapper_manifest.and_then(extract_seed))
            .or_else(|| remote_manifest.and_then(extract_seed))
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.config.seed)
            });
        let preset = explicit_experiment
            .and_then(|experiment| experiment.resolved.as_ref())
            .and_then(|resolved| resolved.preset.clone())
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.preset.clone())
            })
            .or_else(|| remote_manifest.and_then(|manifest| manifest.preset.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.preset.clone()));
        let lane = explicit_experiment
            .and_then(|experiment| experiment.resolved.as_ref())
            .and_then(|resolved| resolved.lane.clone())
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.lane.clone())
            })
            .or_else(|| remote_manifest.and_then(|manifest| manifest.lane.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.lane.clone()));
        let comparison = explicit_experiment
            .and_then(|experiment| experiment.comparison_contract())
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.comparison_contract())
            })
            .or_else(|| remote_manifest.and_then(|manifest| manifest.comparison_contract()));
        let runtime_surface_policy = artifact
            .and_then(|artifact| artifact.manifest.as_ref())
            .and_then(|manifest| manifest.runtime_surface_policy.clone())
            .or_else(|| {
                remote_manifest.and_then(|manifest| manifest.runtime_surface_policy.clone())
            });
        let benchmark_mode = explicit_experiment
            .and_then(|experiment| experiment.benchmark_mode())
            .or_else(|| artifact_experiment.and_then(ArtifactExperimentRecord::benchmark_mode))
            .unwrap_or(BenchmarkMode::Leaderboard);
        let logical_name = explicit_experiment
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.logical_name.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.experiment_id.as_ref())
                    .and_then(|id| id.logical_name.clone())
            });
        let question_summary = explicit_experiment
            .and_then(|experiment| experiment.question.as_ref())
            .and_then(|question| question.summary.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.question.as_ref())
                    .and_then(|question| question.summary.clone())
            });
        let commit_sha = explicit_experiment
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.commit_sha.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.experiment_id.as_ref())
                    .and_then(|id| id.commit_sha.clone())
            })
            .or_else(|| {
                remote_manifest
                    .and_then(|manifest| manifest.build.as_ref())
                    .and_then(|build| build.commit_sha.clone())
            })
            .or_else(|| {
                wrapper_manifest
                    .and_then(|manifest| manifest.build.as_ref())
                    .and_then(|build| build.commit_sha.clone())
            });
        let created_at_unix_ms = explicit_experiment
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.created_at_unix_ms)
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.experiment_id.as_ref())
                    .and_then(|id| id.created_at_unix_ms)
            })
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.generated_at_unix_ms)
            });
        let backend = explicit_experiment
            .and_then(|experiment| experiment.execution.as_ref())
            .and_then(|execution| execution.backend.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.execution.as_ref())
                    .and_then(|execution| execution.backend.clone())
            })
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.backend.clone())
            })
            .or_else(|| remote_manifest.and_then(|manifest| manifest.backend.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.backend.clone()));
        let pod_id = wrapper_manifest
            .and_then(|manifest| manifest.pod.as_ref())
            .and_then(|pod| pod.id.clone())
            .or_else(|| {
                remote_manifest
                    .and_then(|manifest| manifest.pod.as_ref())
                    .and_then(|pod| pod.id.clone())
            });
        let species = explicit_experiment
            .and_then(|experiment| experiment.variant.as_ref())
            .and_then(|variant| variant.species.as_deref())
            .and_then(|species| species.parse::<SpeciesId>().ok())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.variant.as_ref())
                    .and_then(|variant| variant.species.as_deref())
                    .and_then(|species| species.parse::<SpeciesId>().ok())
            })
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.results.first())
                    .and_then(|record| {
                        record
                            .species
                            .as_deref()
                            .and_then(|value| value.parse().ok())
                    })
            });
        let variant_name = explicit_experiment
            .and_then(|experiment| experiment.variant.as_ref())
            .and_then(|variant| variant.variant_name.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.variant.as_ref())
                    .and_then(|variant| variant.variant_name.clone())
            })
            .or_else(|| species.map(species_variant_name));
        let run_id = explicit_experiment
            .and_then(|experiment| experiment.experiment_id.as_ref())
            .and_then(|id| id.run_id.clone())
            .or_else(|| {
                artifact_experiment
                    .and_then(|experiment| experiment.experiment_id.as_ref())
                    .and_then(|id| id.run_id.clone())
            })
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.run_id.clone())
            })
            .or_else(|| remote_manifest.and_then(|manifest| manifest.run_id.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.run_id.clone()))
            .unwrap_or_else(|| {
                run_dir
                    .file_name()
                    .map(|name| name.to_string_lossy().to_string())
                    .unwrap_or_else(|| run_dir.display().to_string())
            });

        Self {
            run_id,
            seed,
            preset,
            lane,
            comparison,
            runtime_surface_policy,
            benchmark_mode,
            logical_name,
            question_summary,
            commit_sha,
            created_at_unix_ms,
            backend,
            pod_id,
            species,
            variant_name,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct CompletedRow {
    run_id: String,
    seed: Option<u64>,
    preset: String,
    lane: Option<String>,
    comparison: ComparisonContract,
    runtime_surface_policy: String,
    benchmark_mode: BenchmarkMode,
    logical_name: Option<String>,
    question_summary: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
    backend: Option<String>,
    variant_name: String,
    species: SpeciesId,
    rank: usize,
    fitness: f64,
    stability: f64,
    perplexity: f64,
    arc: f64,
    tok_s: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct FailureRow {
    run_id: String,
    seed: Option<u64>,
    preset: Option<String>,
    lane: Option<String>,
    comparison: Option<ComparisonContract>,
    runtime_surface_policy: Option<String>,
    benchmark_mode: BenchmarkMode,
    logical_name: Option<String>,
    question_summary: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
    backend: Option<String>,
    variant_name: Option<String>,
    species: Option<SpeciesId>,
    outcome: String,
    error: String,
}

impl FailureRow {
    fn matches_seed_and_contract(&self, key: &RunGroupKey) -> bool {
        self.seed == key.seed.as_option()
            && self.preset.as_deref() == Some(key.preset.as_str())
            && self
                .comparison
                .as_ref()
                .map(|comparison| comparison.label())
                == Some(key.comparison_label.as_str())
            && self.runtime_surface_policy.as_deref() == Some(key.runtime_surface_policy.as_str())
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct PendingRow {
    run_id: String,
    seed: Option<u64>,
    preset: Option<String>,
    lane: Option<String>,
    comparison: Option<ComparisonContract>,
    runtime_surface_policy: Option<String>,
    benchmark_mode: BenchmarkMode,
    logical_name: Option<String>,
    question_summary: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
    backend: Option<String>,
    variant_name: Option<String>,
    species: Option<SpeciesId>,
    status: String,
    pod_id: Option<String>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct AggregateRow {
    variant_name: String,
    samples: usize,
    comparison: ComparisonContract,
    rank: usize,
    fitness: f64,
    stability: f64,
    perplexity: f64,
    arc: f64,
    tok_s: f64,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct RankedResult {
    rank: usize,
    fitness: f64,
    stability: f64,
    perplexity: f64,
    arc: f64,
    tok_s: f64,
}

#[derive(Clone, Debug)]
struct LedgerEntry {
    when_unix_ms: u64,
    when: String,
    run_id: String,
    logical_name: Option<String>,
    question_summary: Option<String>,
    variant_name: String,
    preset: String,
    benchmark_mode: BenchmarkMode,
    runtime_surface_policy: String,
    commit_sha: Option<String>,
    outcome: String,
    fitness: Option<f64>,
    tok_s: Option<f64>,
    change_axis: String,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlManifest {
    run_id: Option<String>,
    experiment: Option<RunControlExperimentRecord>,
    status: Option<String>,
    exit_code: Option<i64>,
    started_at: Option<String>,
    finished_at: Option<String>,
    pod: Option<RunPodRecord>,
    runtime: Option<RunRuntimeRecord>,
    paths: Option<RunPathsRecord>,
    build: Option<RunBuildRecord>,
    comparison_authority: Option<String>,
    comparison_contract: Option<ComparisonContractRecord>,
    runtime_surface_policy: Option<String>,
    preset: Option<String>,
    lane: Option<String>,
    backend: Option<String>,
    execution_mode: Option<String>,
    pod_id: Option<String>,
    timeout_seconds: Option<f64>,
    wrapper_timeout_seconds: Option<u64>,
    config: RunConfigRecord,
}

impl RunControlManifest {
    fn comparison_contract(&self) -> Option<ComparisonContract> {
        parse_comparison_contract(
            self.comparison_contract.as_ref(),
            self.comparison_authority.as_deref(),
        )
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlExperimentRecord {
    experiment_id: Option<RunControlExperimentIdRecord>,
    question: Option<RunControlQuestionRecord>,
    variant: Option<RunControlVariantRecord>,
    runtime: Option<RunControlRuntimeSurfaceRecord>,
    comparison: Option<ComparisonContractRecord>,
    execution: Option<RunControlExecutionRecord>,
    resolved: Option<RunControlResolvedRecord>,
}

impl RunControlExperimentRecord {
    fn comparison_contract(&self) -> Option<ComparisonContract> {
        parse_comparison_contract(self.comparison.as_ref(), None)
    }

    fn benchmark_mode(&self) -> Option<BenchmarkMode> {
        self.runtime
            .as_ref()
            .and_then(|runtime| runtime.benchmark_mode.as_deref())
            .and_then(parse_benchmark_mode_label)
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlExperimentIdRecord {
    logical_name: Option<String>,
    run_id: Option<String>,
    branch: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlQuestionRecord {
    summary: Option<String>,
    lane_intent: Option<String>,
    decision_intent: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlVariantRecord {
    species: Option<String>,
    variant_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlRuntimeSurfaceRecord {
    eval_backend_policy: Option<String>,
    batching_policy: Option<String>,
    execution_policy: Option<String>,
    buffer_reuse_policy: Option<String>,
    benchmark_mode: Option<String>,
    backend_policy: Option<String>,
    label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlExecutionRecord {
    backend: Option<String>,
    execution_mode: Option<String>,
    pod_id: Option<String>,
    wrapper_timeout_seconds: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlResolvedRecord {
    lane: Option<String>,
    preset: Option<String>,
    sequence: Option<String>,
    seed: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunRuntimeRecord {
    backend: Option<String>,
    run_timeout_seconds: Option<u64>,
    tournament_args: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunPodRecord {
    id: Option<String>,
    name: Option<String>,
    status: Option<String>,
    cost_per_hr: Option<f64>,
    gpu_count: Option<u64>,
    image_name: Option<String>,
    volume_in_gb: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunPathsRecord {
    remote_dir: Option<String>,
    state_dir: Option<String>,
    preservation_root: Option<String>,
    manifest_path: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunBuildRecord {
    branch: Option<String>,
    build_key: Option<String>,
    commit_sha: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunConfigRecord {
    seed: Option<u64>,
    dim: Option<usize>,
    levels: Option<usize>,
    vocab_size: Option<usize>,
    max_seq_len: Option<usize>,
    max_recursion_depth: Option<usize>,
    stability_depth: Option<usize>,
    router_threshold: Option<f32>,
    train_batch_size: Option<usize>,
    eval_batch_size: Option<usize>,
    train_steps_per_species: Option<usize>,
    eval_batches_per_family: Option<usize>,
    learning_rate: Option<f64>,
    parallelism: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunArtifactFile {
    manifest: Option<RunManifestFile>,
    results: Vec<ArtifactSpeciesRecord>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunManifestFile {
    run_id: Option<String>,
    generated_at_unix_ms: Option<u64>,
    comparison_authority: Option<String>,
    comparison_contract: Option<ComparisonContractRecord>,
    runtime_surface_policy: Option<String>,
    preset: Option<String>,
    lane: Option<String>,
    backend: Option<String>,
    execution_mode: Option<String>,
    pod_id: Option<String>,
    timeout_seconds: Option<f64>,
    wrapper_timeout_seconds: Option<u64>,
    config: RunConfigRecord,
    experiments: Vec<ArtifactExperimentRecord>,
}

impl RunManifestFile {
    fn comparison_contract(&self) -> Option<ComparisonContract> {
        parse_comparison_contract(
            self.comparison_contract.as_ref(),
            self.comparison_authority.as_deref(),
        )
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ArtifactSpeciesRecord {
    variant_name: Option<String>,
    species: Option<String>,
    ordinal: Option<usize>,
    total: Option<usize>,
    outcome_class: Option<String>,
    execution_outcome: Option<String>,
    quality_outcome: Option<String>,
    comparison_authority: Option<String>,
    runtime_surface_policy: Option<String>,
    experiment: Option<ArtifactExperimentRecord>,
    error: Option<String>,
    timeout_seconds: Option<f64>,
    phase_timings: Vec<ArtifactPhaseTiming>,
    metrics: Option<ArtifactMetricsRecord>,
    ranked_result: Option<ArtifactRankedResult>,
}

impl ArtifactSpeciesRecord {
    fn comparison_contract(&self) -> Option<ComparisonContract> {
        self.experiment
            .as_ref()
            .and_then(|experiment| experiment.comparison_contract())
            .or_else(|| parse_comparison_contract(None, self.comparison_authority.as_deref()))
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ComparisonContractRecord {
    authority: Option<String>,
    requires_same_preset: Option<bool>,
    requires_same_runtime_surfaces: Option<bool>,
    requires_frozen_commit: Option<bool>,
    requires_same_backend: Option<bool>,
    label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ArtifactExperimentRecord {
    experiment_id: Option<ArtifactExperimentIdRecord>,
    question: Option<ArtifactQuestionRecord>,
    variant: Option<RunControlVariantRecord>,
    runtime: Option<ArtifactRuntimeSurfaceRecord>,
    comparison: Option<ComparisonContractRecord>,
    execution: Option<ArtifactExecutionRecord>,
}

impl ArtifactExperimentRecord {
    fn comparison_contract(&self) -> Option<ComparisonContract> {
        parse_comparison_contract(self.comparison.as_ref(), None)
    }

    fn benchmark_mode(&self) -> Option<BenchmarkMode> {
        self.runtime
            .as_ref()
            .and_then(|runtime| runtime.benchmark_mode.as_deref())
            .and_then(parse_benchmark_mode_label)
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
struct ArtifactExperimentIdRecord {
    logical_name: Option<String>,
    run_id: Option<String>,
    branch: Option<String>,
    commit_sha: Option<String>,
    created_at_unix_ms: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
struct ArtifactQuestionRecord {
    summary: Option<String>,
    lane_intent: Option<String>,
    decision_intent: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
struct ArtifactRuntimeSurfaceRecord {
    eval_backend_policy: Option<String>,
    batching_policy: Option<String>,
    execution_policy: Option<String>,
    buffer_reuse_policy: Option<String>,
    benchmark_mode: Option<String>,
    backend_policy: Option<String>,
    label: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
struct ArtifactExecutionRecord {
    kind: Option<String>,
    backend: Option<String>,
    execution_mode: Option<String>,
    pod_id: Option<String>,
    wrapper_timeout_seconds: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ArtifactPhaseTiming {
    phase: Option<String>,
    elapsed_seconds: Option<f64>,
    completed: Option<usize>,
    total: Option<usize>,
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ArtifactMetricsRecord {
    grad_norm_depth_20: Option<f64>,
    long_context_perplexity: Option<f64>,
    arc_accuracy: Option<f64>,
    tokens_per_sec: Option<f64>,
}

impl ArtifactMetricsRecord {
    fn to_species_raw_metrics(&self, species: SpeciesId) -> Option<SpeciesRawMetrics> {
        Some(SpeciesRawMetrics {
            species,
            grad_norm_depth_20: self.grad_norm_depth_20?,
            long_context_perplexity: self.long_context_perplexity?,
            arc_accuracy: self.arc_accuracy?,
            tokens_per_sec: self.tokens_per_sec?,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct ArtifactRankedResult {
    rank: Option<usize>,
    fitness: Option<f64>,
    stability_score: Option<f64>,
    long_context_perplexity: Option<f64>,
    arc_accuracy: Option<f64>,
    tokens_per_sec: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[allow(dead_code)]
struct RunGroupKey {
    seed: SeedLabel,
    preset: String,
    comparison_label: String,
    runtime_surface_policy: String,
    lane: Option<String>,
}

impl RunGroupKey {
    fn from_row(row: &CompletedRow) -> Self {
        Self {
            seed: SeedLabel::from_option(row.seed),
            preset: row.preset.clone(),
            comparison_label: row.comparison.label().to_owned(),
            runtime_surface_policy: row.runtime_surface_policy.clone(),
            lane: row.lane.clone(),
        }
    }

    fn seed_label(&self) -> String {
        self.seed.label()
    }

    fn comparison_label(&self) -> &str {
        &self.comparison_label
    }
}

fn parse_comparison_contract(
    record: Option<&ComparisonContractRecord>,
    legacy_label: Option<&str>,
) -> Option<ComparisonContract> {
    if let Some(record) = record {
        return Some(ComparisonContract {
            authority: match record.authority.as_deref()? {
                "authoritative" => ComparisonAuthority::Authoritative,
                "advisory" => ComparisonAuthority::Advisory,
                _ => return None,
            },
            requires_same_preset: record.requires_same_preset.unwrap_or(false),
            requires_same_runtime_surfaces: record.requires_same_runtime_surfaces.unwrap_or(true),
            requires_frozen_commit: record.requires_frozen_commit.unwrap_or(false),
            requires_same_backend: record.requires_same_backend.unwrap_or(true),
        });
    }

    match legacy_label {
        Some("authoritative same-preset") => Some(ComparisonContract::authoritative_same_preset()),
        Some("advisory mixed-preset") => Some(ComparisonContract::advisory_mixed_preset()),
        Some("advisory same-preset") => Some(ComparisonContract {
            authority: ComparisonAuthority::Advisory,
            requires_same_preset: true,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: false,
            requires_same_backend: true,
        }),
        Some("authoritative mixed-preset") => Some(ComparisonContract {
            authority: ComparisonAuthority::Authoritative,
            requires_same_preset: false,
            requires_same_runtime_surfaces: true,
            requires_frozen_commit: true,
            requires_same_backend: true,
        }),
        _ => None,
    }
}

fn extract_seed(manifest: &RunControlManifest) -> Option<u64> {
    manifest.config.seed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write};

    fn write_json(path: &Path, value: serde_json::Value) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let mut file = fs::File::create(path).unwrap();
        write!(file, "{}", serde_json::to_string_pretty(&value).unwrap()).unwrap();
    }

    #[test]
    fn summarize_root_reads_completed_runs_and_pending_runs() {
        let root =
            std::env::temp_dir().join(format!("fractal-bakeoff-summary-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let completed = root.join("logical-winner").join("20260331T000000Z_seed42");
        write_json(
            &completed.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "completed-run",
                "pod": { "id": "pod-42", "name": "fractal-winner-bakeoff-s42-a100" },
                "experiment": {
                    "experiment_id": {
                        "logical_name": "winner-bakeoff",
                        "run_id": "completed-run",
                        "commit_sha": "abc123",
                        "created_at_unix_ms": 123
                    },
                    "question": {
                        "summary": "compare winner-lane variants"
                    },
                    "variant": {
                        "species": "p1_contractive",
                        "variant_name": "p1_contractive_v1"
                    },
                    "runtime": {
                        "benchmark_mode": "leaderboard",
                        "backend": "cuda",
                        "label": "authoritative-same-preset"
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "execution": {
                        "backend": "cuda"
                    },
                    "resolved": {
                        "lane": "leader",
                        "preset": "full-medium-stress",
                        "seed": 42
                    }
                }
            }),
        );
        write_json(
            &completed.join("artifacts/tournament-run-artifact.json"),
            serde_json::json!({
                "manifest": {
                    "run_id": "completed-run",
                    "comparison_authority": "authoritative same-preset",
                    "comparison_contract": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "runtime_surface_policy": "conservative-defaults",
                    "preset": "full-medium-stress",
                    "lane": "leader",
                    "config": { "seed": 42 }
                },
                "results": [
                    {
                        "variant_name": "p1_contractive_v1",
                        "species": "p1_contractive",
                        "outcome_class": "success",
                        "comparison_authority": "authoritative same-preset",
                        "runtime_surface_policy": "conservative-defaults",
                        "metrics": {
                            "grad_norm_depth_20": 0.53,
                            "long_context_perplexity": 1.54,
                            "arc_accuracy": 0.68,
                            "tokens_per_sec": 100.0
                        },
                        "ranked_result": {
                            "rank": 1,
                            "fitness": 0.58,
                            "stability_score": 0.53,
                            "long_context_perplexity": 1.54,
                            "arc_accuracy": 0.68,
                            "tokens_per_sec": 100.0
                        }
                    }
                ]
            }),
        );

        let pending = root.join("logical-winner").join("20260331T000001Z_seed43");
        write_json(
            &pending.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "pending-run",
                "status": "running",
                "pod": { "id": "pod-43", "name": "fractal-winner-bakeoff-s43-a100" },
                "experiment": {
                    "experiment_id": {
                        "logical_name": "winner-bakeoff",
                        "run_id": "pending-run",
                        "commit_sha": "abc123",
                        "created_at_unix_ms": 456
                    },
                    "question": {
                        "summary": "compare winner-lane variants"
                    },
                    "variant": {
                        "species": "p1_fractal_hybrid",
                        "variant_name": "p1_fractal_hybrid_v1"
                    },
                    "runtime": {
                        "benchmark_mode": "leaderboard",
                        "backend": "cuda",
                        "label": "authoritative-same-preset"
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "execution": {
                        "backend": "cuda"
                    },
                    "resolved": {
                        "lane": "leader",
                        "preset": "full-medium-stress",
                        "seed": 43
                    }
                }
            }),
        );

        let output = summarize_root(&root, ReportKind::Leaderboard).unwrap();
        assert!(output.contains("Per-seed leaderboards:"));
        assert!(output.contains("p1_contractive_v1"));
        assert!(output.contains("Aggregate authoritative leaderboard:"));
        assert!(output.contains("runtime=conservative-defaults"));
        assert!(output.contains("Pending runs:"));
        assert!(output.contains("p1_fractal_hybrid_v1"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn run_group_key_keeps_seed_and_authority() {
        let row = CompletedRow {
            run_id: "run-a".to_owned(),
            seed: Some(42),
            preset: "full-medium-stress".to_owned(),
            lane: Some("leader".to_owned()),
            comparison: ComparisonContract::authoritative_same_preset(),
            runtime_surface_policy: "conservative-defaults".to_owned(),
            benchmark_mode: BenchmarkMode::Leaderboard,
            logical_name: Some("winner-bakeoff".to_owned()),
            question_summary: Some("compare winner-lane variants".to_owned()),
            commit_sha: Some("abc123".to_owned()),
            created_at_unix_ms: Some(123),
            backend: Some("cuda".to_owned()),
            variant_name: "p1_contractive_v1".to_owned(),
            species: SpeciesId::P1Contractive,
            rank: 1,
            fitness: 0.58,
            stability: 0.53,
            perplexity: 1.54,
            arc: 0.68,
            tok_s: 100.0,
        };
        let key = RunGroupKey::from_row(&row);
        assert_eq!(key.seed_label(), "42");
        assert_eq!(key.comparison_label(), "authoritative same-preset");
    }

    #[test]
    fn summarize_root_supports_systems_speed_report() {
        let root = std::env::temp_dir().join(format!(
            "fractal-systems-speed-summary-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);

        let completed = root.join("logical-speed").join("20260331T000002Z_speed42");
        write_json(
            &completed.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "speed-run",
                "experiment": {
                    "experiment_id": {
                        "logical_name": "systems-speed-p1-fractal-hybrid",
                        "run_id": "speed-run",
                        "commit_sha": "abc123",
                        "created_at_unix_ms": 123
                    },
                    "question": {
                        "summary": "measure systems throughput only"
                    },
                    "variant": {
                        "species": "p1_fractal_hybrid",
                        "variant_name": "p1_fractal_hybrid_v1"
                    },
                    "runtime": {
                        "benchmark_mode": "systems-speed",
                        "backend": "cuda",
                        "label": "systems-speed"
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "execution": {
                        "backend": "cuda"
                    },
                    "resolved": {
                        "lane": "leader",
                        "preset": "full-medium-stress",
                        "seed": 42
                    }
                }
            }),
        );
        write_json(
            &completed.join("artifacts/tournament-run-artifact.json"),
            serde_json::json!({
                "manifest": {
                    "run_id": "speed-run",
                    "runtime_surface_policy": "eval_backend=shared_backend batching=padded execution=simple-loop buffer_reuse=disabled benchmark_mode=systems-speed backend_policy=active-execution-backend",
                    "preset": "full-medium-stress",
                    "lane": "leader",
                    "config": { "seed": 42 }
                },
                "results": [
                    {
                        "variant_name": "p1_fractal_hybrid_v1",
                        "species": "p1_fractal_hybrid",
                        "outcome_class": "success",
                        "experiment": {
                            "experiment_id": {
                                "logical_name": "systems-speed-p1-fractal-hybrid",
                                "commit_sha": "abc123",
                                "created_at_unix_ms": 123
                            },
                            "question": {
                                "summary": "measure systems throughput only"
                            },
                            "runtime": {
                                "benchmark_mode": "systems-speed"
                            }
                        },
                        "ranked_result": {
                            "rank": 1,
                            "fitness": 0.55,
                            "stability_score": 0.45,
                            "long_context_perplexity": 1.60,
                            "arc_accuracy": 0.70,
                            "tokens_per_sec": 321.0
                        }
                    }
                ]
            }),
        );

        let output = summarize_root(&root, ReportKind::SystemsSpeed).unwrap();
        assert!(output.contains("Systems-Speed Summary"));
        assert!(output.contains("Aggregate systems-speed leaderboard:"));
        assert!(output.contains("p1_fractal_hybrid_v1"));
        assert!(output.contains("321"));

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn summarize_root_supports_ledger_report() {
        let root =
            std::env::temp_dir().join(format!("fractal-ledger-summary-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);

        let completed = root
            .join("logical-ledger")
            .join("20260331T000003Z_ledger42");
        write_json(
            &completed.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "ledger-run",
                "experiment": {
                    "experiment_id": {
                        "logical_name": "winner-bakeoff-p1-contractive",
                        "run_id": "ledger-run",
                        "commit_sha": "def456",
                        "created_at_unix_ms": 456
                    },
                    "question": {
                        "summary": "rerun frozen winner benchmark"
                    },
                    "variant": {
                        "species": "p1_contractive",
                        "variant_name": "p1_contractive_v1"
                    },
                    "runtime": {
                        "benchmark_mode": "leaderboard",
                        "backend": "cuda",
                        "label": "authoritative-same-preset"
                    },
                    "comparison": {
                        "authority": "authoritative",
                        "requires_same_preset": true,
                        "requires_same_runtime_surfaces": true,
                        "requires_frozen_commit": true,
                        "requires_same_backend": true
                    },
                    "execution": {
                        "backend": "cuda"
                    },
                    "resolved": {
                        "lane": "leader",
                        "preset": "full-medium-stress",
                        "seed": 42
                    }
                }
            }),
        );
        write_json(
            &completed.join("artifacts/tournament-run-artifact.json"),
            serde_json::json!({
                "manifest": {
                    "run_id": "ledger-run",
                    "preset": "full-medium-stress",
                    "lane": "leader",
                    "config": { "seed": 42 }
                },
                "results": [
                    {
                        "variant_name": "p1_contractive_v1",
                        "species": "p1_contractive",
                        "outcome_class": "success",
                        "experiment": {
                            "experiment_id": {
                                "logical_name": "winner-bakeoff-p1-contractive",
                                "commit_sha": "def456",
                                "created_at_unix_ms": 456
                            },
                            "question": {
                                "summary": "rerun frozen winner benchmark"
                            },
                            "runtime": {
                                "benchmark_mode": "leaderboard"
                            }
                        },
                        "ranked_result": {
                            "rank": 1,
                            "fitness": 0.65,
                            "stability_score": 0.45,
                            "long_context_perplexity": 1.54,
                            "arc_accuracy": 0.79,
                            "tokens_per_sec": 5.0
                        }
                    }
                ]
            }),
        );

        let output = summarize_root(&root, ReportKind::Ledger).unwrap();
        assert!(output.contains("Experiment Ledger"));
        assert!(output.contains("winner-bakeoff-p1-contractive"));
        assert!(output.contains("rerun frozen winner benchmark"));
        assert!(output.contains("success"));

        let _ = fs::remove_dir_all(&root);
    }
}
