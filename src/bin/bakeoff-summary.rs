use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use fractal::{species_registry_for_species, ComparisonAuthority, SpeciesId, SpeciesRawMetrics};
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
    let output = summarize_root(&args.root)?;
    print!("{output}");
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CliArgs {
    root: PathBuf,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut root = PathBuf::from(DEFAULT_ROOT);
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

        Ok(Self { root })
    }
}

fn usage() -> String {
    let mut output = String::new();
    let _ = writeln!(
        output,
        "Usage: cargo run --bin bakeoff-summary -- [--root <path>]"
    );
    let _ = writeln!(output, "Default root: {DEFAULT_ROOT}");
    output
}

fn summarize_root(root: &Path) -> Result<String, String> {
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

    let completed = dedupe_completed_rows(
        runs.iter()
            .flat_map(|run| run.completed_rows.iter().cloned())
            .collect(),
    );
    let failures: Vec<_> = runs
        .iter()
        .flat_map(|run| run.failed_rows.iter().cloned())
        .collect();
    let pending: Vec<_> = runs
        .iter()
        .flat_map(|run| run.pending_rows.iter().cloned())
        .collect();

    let mut output = String::new();
    writeln!(output, "Bakeoff Summary").unwrap();
    writeln!(output, "root: {}", scanned_root.display()).unwrap();
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
                "seed={} | preset={} | authority={}{}",
                group_key.seed_label(),
                group_key.preset,
                group_key.authority_label(),
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
            .filter(|row| row.authority == ComparisonAuthority::AuthoritativeSamePreset)
            .cloned()
            .collect(),
    );
    let advisory_aggregates = aggregate_rows(
        completed
            .iter()
            .filter(|row| row.authority == ComparisonAuthority::AdvisoryMixedPreset)
            .cloned()
            .collect(),
    );

    writeln!(output, "Aggregate authoritative leaderboard:").unwrap();
    if authoritative_aggregates.is_empty() {
        writeln!(output, "(none yet)").unwrap();
    } else {
        for (preset, rows) in authoritative_aggregates {
            writeln!(output, "preset={preset}").unwrap();
            output.push_str(&render_aggregate_table(&rows));
            writeln!(output).unwrap();
        }
    }

    if !advisory_aggregates.is_empty() {
        writeln!(output, "Advisory snapshot:").unwrap();
        for (preset, rows) in advisory_aggregates {
            writeln!(output, "preset={preset}").unwrap();
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
                        authority: metadata.authority,
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
                    authority: metadata.authority,
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
                authority: metadata.authority,
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
                authority: metadata.authority,
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
            authority: metadata.authority,
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
        authority: metadata
            .authority
            .unwrap_or(ComparisonAuthority::AuthoritativeSamePreset),
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

fn dedupe_completed_rows(mut rows: Vec<CompletedRow>) -> Vec<CompletedRow> {
    let mut best_by_key: BTreeMap<(Option<u64>, String, String, String), CompletedRow> =
        BTreeMap::new();
    for row in rows.drain(..) {
        let key = (
            row.seed,
            row.preset.clone(),
            row.variant_name.clone(),
            row.authority.label().to_owned(),
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

fn aggregate_rows(rows: Vec<CompletedRow>) -> BTreeMap<String, Vec<AggregateRow>> {
    let mut grouped: BTreeMap<(String, String, String), Vec<CompletedRow>> = BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                row.preset.clone(),
                row.authority.label().to_owned(),
                row.variant_name.clone(),
            ))
            .or_default()
            .push(row);
    }

    let mut output: BTreeMap<String, Vec<AggregateRow>> = BTreeMap::new();
    for ((preset, authority_label, variant_name), entries) in grouped {
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

        output.entry(preset).or_default().push(AggregateRow {
            variant_name,
            samples: entries.len(),
            rank,
            authority: parse_authority(&authority_label)
                .unwrap_or(ComparisonAuthority::AuthoritativeSamePreset),
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
                row.authority.label().to_owned(),
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
    authority: Option<ComparisonAuthority>,
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
        let seed = wrapper_manifest
            .and_then(extract_seed)
            .or_else(|| remote_manifest.and_then(extract_seed))
            .or_else(|| {
                artifact
                    .and_then(|artifact| artifact.manifest.as_ref())
                    .and_then(|manifest| manifest.config.seed)
            });
        let preset = artifact
            .and_then(|artifact| artifact.manifest.as_ref())
            .and_then(|manifest| manifest.preset.clone())
            .or_else(|| remote_manifest.and_then(|manifest| manifest.preset.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.preset_from_args()));
        let lane = artifact
            .and_then(|artifact| artifact.manifest.as_ref())
            .and_then(|manifest| manifest.lane.clone())
            .or_else(|| remote_manifest.and_then(|manifest| manifest.lane.clone()))
            .or_else(|| wrapper_manifest.and_then(|manifest| manifest.lane_from_args()));
        let authority = artifact
            .and_then(|artifact| artifact.manifest.as_ref())
            .and_then(|manifest| manifest.comparison_authority.as_deref())
            .and_then(parse_authority)
            .or_else(|| {
                remote_manifest
                    .and_then(|manifest| manifest.comparison_authority.as_deref())
                    .and_then(parse_authority)
            });
        let pod_id = wrapper_manifest
            .and_then(|manifest| manifest.pod.as_ref())
            .and_then(|pod| pod.id.clone())
            .or_else(|| {
                remote_manifest
                    .and_then(|manifest| manifest.pod.as_ref())
                    .and_then(|pod| pod.id.clone())
            });
        let species = wrapper_manifest
            .and_then(|manifest| manifest.species_from_args())
            .or_else(|| remote_manifest.and_then(|manifest| manifest.species_from_args()))
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
        let variant_name = species.map(species_variant_name);
        let run_id = artifact
            .and_then(|artifact| artifact.manifest.as_ref())
            .and_then(|manifest| manifest.run_id.clone())
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
            authority,
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
    authority: ComparisonAuthority,
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
    authority: Option<ComparisonAuthority>,
    variant_name: Option<String>,
    species: Option<SpeciesId>,
    outcome: String,
    error: String,
}

impl FailureRow {
    fn matches_seed_and_contract(&self, key: &RunGroupKey) -> bool {
        self.seed == key.seed.as_option()
            && self.preset.as_deref() == Some(key.preset.as_str())
            && self.authority.as_ref().map(|authority| authority.label())
                == Some(key.authority_label.as_str())
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct PendingRow {
    run_id: String,
    seed: Option<u64>,
    preset: Option<String>,
    lane: Option<String>,
    authority: Option<ComparisonAuthority>,
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
    authority: ComparisonAuthority,
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

#[derive(Clone, Debug, Deserialize, Default)]
#[serde(default)]
#[allow(dead_code)]
struct RunControlManifest {
    run_id: Option<String>,
    status: Option<String>,
    exit_code: Option<i64>,
    started_at: Option<String>,
    finished_at: Option<String>,
    pod: Option<RunPodRecord>,
    runtime: Option<RunRuntimeRecord>,
    paths: Option<RunPathsRecord>,
    build: Option<RunBuildRecord>,
    comparison_authority: Option<String>,
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
    fn preset_from_args(&self) -> Option<String> {
        self.runtime
            .as_ref()
            .and_then(|runtime| extract_arg(&runtime.tournament_args, "--preset"))
    }

    fn lane_from_args(&self) -> Option<String> {
        self.runtime
            .as_ref()
            .and_then(|runtime| extract_arg(&runtime.tournament_args, "--lane"))
    }

    fn species_from_args(&self) -> Option<SpeciesId> {
        self.runtime
            .as_ref()
            .and_then(|runtime| extract_arg(&runtime.tournament_args, "--species"))
            .and_then(|species| species.parse().ok())
    }
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
    comparison_authority: Option<String>,
    preset: Option<String>,
    lane: Option<String>,
    backend: Option<String>,
    execution_mode: Option<String>,
    pod_id: Option<String>,
    timeout_seconds: Option<f64>,
    wrapper_timeout_seconds: Option<u64>,
    config: RunConfigRecord,
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
    error: Option<String>,
    timeout_seconds: Option<f64>,
    phase_timings: Vec<ArtifactPhaseTiming>,
    metrics: Option<ArtifactMetricsRecord>,
    ranked_result: Option<ArtifactRankedResult>,
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
    authority_label: String,
    lane: Option<String>,
}

impl RunGroupKey {
    fn from_row(row: &CompletedRow) -> Self {
        Self {
            seed: SeedLabel::from_option(row.seed),
            preset: row.preset.clone(),
            authority_label: row.authority.label().to_owned(),
            lane: row.lane.clone(),
        }
    }

    fn seed_label(&self) -> String {
        self.seed.label()
    }

    fn authority_label(&self) -> &'static str {
        if self.authority_label == ComparisonAuthority::AuthoritativeSamePreset.label() {
            ComparisonAuthority::AuthoritativeSamePreset.label()
        } else {
            ComparisonAuthority::AdvisoryMixedPreset.label()
        }
    }
}

fn parse_authority(value: &str) -> Option<ComparisonAuthority> {
    match value {
        "authoritative same-preset" => Some(ComparisonAuthority::AuthoritativeSamePreset),
        "advisory mixed-preset" => Some(ComparisonAuthority::AdvisoryMixedPreset),
        _ => None,
    }
}

fn extract_seed(manifest: &RunControlManifest) -> Option<u64> {
    manifest.config.seed
}

fn extract_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find_map(|window| {
        if window[0] == flag {
            Some(window[1].clone())
        } else {
            None
        }
    })
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

        let completed = root.join("20260331T000000Z_seed42");
        write_json(
            &completed.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "completed-run",
                "runtime": { "tournament_args": ["--preset", "full-medium-stress", "--species", "p1_contractive", "--seed", "42"] },
                "pod": { "id": "pod-42", "name": "fractal-winner-bakeoff-s42-a100" }
            }),
        );
        write_json(
            &completed.join("artifacts/tournament-run-artifact.json"),
            serde_json::json!({
                "manifest": {
                    "run_id": "completed-run",
                    "comparison_authority": "authoritative same-preset",
                    "preset": "full-medium-stress",
                    "lane": "leader",
                    "config": { "seed": 42 }
                },
                "results": [
                    {
                        "variant_name": "p1_contractive_v1",
                        "species": "p1_contractive",
                        "outcome_class": "success",
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

        let pending = root.join("20260331T000001Z_seed43");
        write_json(
            &pending.join("metadata/wrapper-manifest.json"),
            serde_json::json!({
                "run_id": "pending-run",
                "status": "running",
                "runtime": { "tournament_args": ["--preset", "full-medium-stress", "--species", "p1_fractal_hybrid", "--seed", "43"] },
                "pod": { "id": "pod-43", "name": "fractal-winner-bakeoff-s43-a100" }
            }),
        );

        let output = summarize_root(&root).unwrap();
        assert!(output.contains("Per-seed leaderboards:"));
        assert!(output.contains("p1_contractive_v1"));
        assert!(output.contains("Aggregate authoritative leaderboard:"));
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
            authority: ComparisonAuthority::AuthoritativeSamePreset,
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
        assert_eq!(key.authority_label(), "authoritative same-preset");
    }
}
