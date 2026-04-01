use std::{fmt, num::NonZeroUsize, path::Path};

use burn::tensor::{
    backend::{AutodiffBackend, Backend},
    Int, Tensor, TensorData,
};
use fractal_core::{
    data_generator::{
        GeneratorConfig, SimpleHierarchicalGenerator, TaskFamily, TokenBatch, PAD_TOKEN,
    },
    error::FractalError,
    lifecycle::{
        ArcSourceMode, ExperimentSpec, TokenizerArtifactSpec, TrainingInputMode, TrainingInputSpec,
    },
    registry::{run_species_with_batches, PrimitiveVariantName, SpeciesId, TrainingBatchSet},
    SpeciesRawMetrics,
};
use fractal_tokenizer::{
    EmbeddingBridgeAdapter, FaceoffChunkLimits, FaceoffTokenizer, FaceoffVocabConfig,
    ModelFacingBatch, ModelFacingDocument, NativeCollationSpec, NativeCompatibilityAdapter,
    NativeCompatibilityError, NativeTokenizer, NativeTruncationPolicy, TypedEmbeddingBridge,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerTrainingCorpus {
    pub name: String,
    pub train_documents: Vec<String>,
    pub eval_documents: Vec<String>,
}

impl TokenizerTrainingCorpus {
    pub fn new(
        name: impl Into<String>,
        train_documents: Vec<String>,
        eval_documents: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            train_documents,
            eval_documents,
        }
    }

    pub fn validate(&self) -> Result<(), FractalError> {
        if self.name.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "tokenizer training corpus name must be non-empty".into(),
            ));
        }
        if self.train_documents.is_empty() {
            return Err(FractalError::InvalidConfig(
                "tokenizer training corpus must include at least one train document".into(),
            ));
        }
        if self.eval_documents.is_empty() {
            return Err(FractalError::InvalidConfig(
                "tokenizer training corpus must include at least one eval document".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerBridgeStats {
    pub corpus_name: String,
    pub tokenizer_artifact_id: String,
    pub training_input_mode: TrainingInputMode,
    pub bridge_enabled: bool,
    pub bridge_observational_only: bool,
    pub arc_source_mode: ArcSourceMode,
    pub train_documents: usize,
    pub eval_documents: usize,
    pub model_facing_documents: usize,
    pub bridge_documents: usize,
    pub bridge_chunks: usize,
    pub bridge_tokens: usize,
    pub native_documents: usize,
    pub native_chunks: usize,
    pub native_tokens: usize,
    pub train_batches: usize,
    pub eval_batches: usize,
    pub sequence_len: usize,
}

impl TokenizerBridgeStats {
    fn new(
        training_input: &TrainingInputSpec,
        corpus: &TokenizerTrainingCorpus,
        model_facing_documents: usize,
        bridge_documents: usize,
        bridge_chunks: usize,
        bridge_tokens: usize,
        native_documents: usize,
        native_chunks: usize,
        native_tokens: usize,
        train_batches: usize,
        eval_batches: usize,
        sequence_len: usize,
    ) -> Result<Self, FractalError> {
        let tokenizer = training_input.tokenizer.as_ref().ok_or_else(|| {
            FractalError::InvalidState(
                "tokenizer-backed training requires tokenizer metadata to compute stats".into(),
            )
        })?;

        Ok(Self {
            corpus_name: corpus.name.clone(),
            tokenizer_artifact_id: tokenizer.artifact_id.clone(),
            training_input_mode: training_input.mode,
            bridge_enabled: training_input.bridge.enabled,
            bridge_observational_only: training_input.bridge.observational_only,
            arc_source_mode: training_input.arc_source.mode,
            train_documents: corpus.train_documents.len(),
            eval_documents: corpus.eval_documents.len(),
            model_facing_documents,
            bridge_documents,
            bridge_chunks,
            bridge_tokens,
            native_documents,
            native_chunks,
            native_tokens,
            train_batches,
            eval_batches,
            sequence_len,
        })
    }
}

#[derive(Clone, Debug)]
pub struct TokenizerTrainingRuntime<N> {
    pub tokenizer: N,
    pub faceoff_vocab_config: FaceoffVocabConfig,
    pub chunk_limits: FaceoffChunkLimits,
    pub native_truncation_policy: NativeTruncationPolicy,
}

impl<N> TokenizerTrainingRuntime<N> {
    pub fn new(tokenizer: N, max_sequence_len: usize) -> Result<Self, FractalError> {
        let max_sequence_len = NonZeroUsize::new(max_sequence_len).ok_or_else(|| {
            FractalError::InvalidConfig("max_sequence_len must be greater than zero".into())
        })?;
        Ok(Self {
            tokenizer,
            faceoff_vocab_config: FaceoffVocabConfig::default(),
            chunk_limits: FaceoffChunkLimits::new(max_sequence_len.get(), max_sequence_len.get()),
            native_truncation_policy: NativeTruncationPolicy::Reject,
        })
    }
}

pub fn load_native_tokenizer(
    tokenizer: &TokenizerArtifactSpec,
) -> Result<fractal_tokenizer::HuggingFaceNativeTokenizer, FractalError> {
    let path = tokenizer.artifact_path.as_ref().ok_or_else(|| {
        FractalError::InvalidConfig(
            "tokenizer-backed text training requires a tokenizer artifact path".into(),
        )
    })?;
    fractal_tokenizer::HuggingFaceNativeTokenizer::from_file(Path::new(path))
        .map_err(|error| FractalError::InvalidState(error.to_string()))
}

pub fn build_tokenizer_backed_batches<B, N>(
    config: &fractal_core::TournamentConfig,
    training_input: &TrainingInputSpec,
    corpus: &TokenizerTrainingCorpus,
    runtime: &TokenizerTrainingRuntime<N>,
    device: &B::Device,
) -> Result<(TrainingBatchSet<B>, TokenizerBridgeStats), FractalError>
where
    B: AutodiffBackend,
    N: NativeTokenizer<Token = u32>,
    N::Error: fmt::Display + fmt::Debug + Send + Sync + 'static,
{
    training_input.validate_against_config(config)?;
    corpus.validate()?;
    if training_input.mode != TrainingInputMode::TokenizerBackedText {
        return Err(FractalError::InvalidConfig(
            "tokenizer-backed batch construction requires tokenizer-backed text input mode".into(),
        ));
    }

    if let Some(expected_corpus) = &training_input.corpus_name {
        if expected_corpus != &corpus.name {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer-backed corpus name {} does not match experiment corpus {}",
                corpus.name, expected_corpus
            )));
        }
    }

    let faceoff_config = fractal_tokenizer::TokenizerConfig {
        dim: config.dim,
        levels: config.levels,
        max_depth: config.max_recursion_depth,
        seed: config.seed,
        split_policy: fractal_tokenizer::SplitPolicy::Balanced,
        substrate_mode: fractal_tokenizer::TokenizerSubstrateMode::RawBytes,
    };
    let faceoff_tokenizer = FaceoffTokenizer::new(faceoff_config);

    let all_documents = corpus
        .train_documents
        .iter()
        .chain(corpus.eval_documents.iter())
        .map(String::as_str)
        .collect::<Vec<_>>();
    let vocab = faceoff_tokenizer
        .induce_vocab_from_texts::<B>(&all_documents, device)
        .map_err(|error| FractalError::InvalidState(error.to_string()))?;

    let train = build_split_batches::<B, N>(
        &faceoff_tokenizer,
        &vocab,
        &corpus.train_documents,
        config.train_batch_size,
        training_input,
        &runtime.tokenizer,
        &runtime.chunk_limits,
        &runtime.native_truncation_policy,
        device,
    )?;
    let eval = build_split_batches::<B, N>(
        &faceoff_tokenizer,
        &vocab,
        &corpus.eval_documents,
        config.eval_batch_size,
        training_input,
        &runtime.tokenizer,
        &runtime.chunk_limits,
        &runtime.native_truncation_policy,
        device,
    )?;

    let arc_batches = build_canonical_arc_batches::<B>(config, device)?;

    let stats = TokenizerBridgeStats::new(
        training_input,
        corpus,
        train.model_facing_documents + eval.model_facing_documents,
        train.bridge_documents + eval.bridge_documents,
        train.bridge_chunks + eval.bridge_chunks,
        train.bridge_tokens + eval.bridge_tokens,
        train.native_documents + eval.native_documents,
        train.native_chunks + eval.native_chunks,
        train.native_tokens + eval.native_tokens,
        train.batches.len(),
        eval.batches.len(),
        std::cmp::max(train.sequence_len, eval.sequence_len),
    )?;

    Ok((
        TrainingBatchSet {
            train_sentence: train.batches.clone(),
            train_arc: arc_batches.train_arc,
            eval_sentence: eval.batches.clone(),
            eval_arc: arc_batches.eval_arc,
        },
        stats,
    ))
}

pub fn run_tokenizer_backed_species<B, N>(
    species: SpeciesId,
    variant_name: PrimitiveVariantName,
    config: fractal_core::TournamentConfig,
    experiment: Option<ExperimentSpec>,
    corpus: &TokenizerTrainingCorpus,
    runtime: &TokenizerTrainingRuntime<N>,
    device: B::Device,
) -> Result<(SpeciesRawMetrics, TokenizerBridgeStats), FractalError>
where
    B: AutodiffBackend,
    N: NativeTokenizer<Token = u32>,
    N::Error: fmt::Display + fmt::Debug + Send + Sync + 'static,
{
    let experiment = experiment.or_else(|| config.resolved_experiment(species, variant_name));
    let training_input = experiment
        .as_ref()
        .map(|spec| &spec.training_input)
        .ok_or_else(|| {
            FractalError::InvalidConfig(
                "tokenizer-backed training requires an experiment spec".into(),
            )
        })?;
    let (batches, stats) =
        build_tokenizer_backed_batches::<B, N>(&config, training_input, corpus, runtime, &device)?;

    let metrics = match species {
        SpeciesId::P1Contractive => {
            let rule = fractal_primitives_private::P1Contractive::<B>::new(config.dim, &device);
            run_species_with_batches::<B, _>(
                species,
                variant_name,
                config,
                experiment,
                device,
                rule,
                batches,
            )?
        }
        SpeciesId::P1FractalHybrid => {
            let rule = fractal_primitives_private::P1FractalHybrid::<B>::new(config.dim, &device);
            run_species_with_batches::<B, _>(
                species,
                variant_name,
                config,
                experiment,
                device,
                rule,
                batches,
            )?
        }
        SpeciesId::P1FractalHybridComposite => {
            let rule =
                fractal_primitives_private::P1FractalHybridComposite::<B>::new(config.dim, &device);
            run_species_with_batches::<B, _>(
                species,
                variant_name,
                config,
                experiment,
                device,
                rule,
                batches,
            )?
        }
        _ => {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer-backed Stage 0 only supports the locked top cohort, got {}",
                species
            )));
        }
    };

    Ok((metrics, stats))
}

fn build_canonical_arc_batches<B: AutodiffBackend>(
    config: &fractal_core::TournamentConfig,
    device: &B::Device,
) -> Result<TrainingBatchSet<B>, FractalError> {
    let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        train_examples_per_family: config.train_batch_size.max(1),
        eval_examples_per_family: config.eval_batches_per_family.max(1),
        seed: config.seed,
        depth_config: config.generator_depth_config,
    })?;

    Ok(TrainingBatchSet {
        train_sentence: Vec::new(),
        train_arc: generator.train_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.train_batch_size,
            device,
        )?,
        eval_sentence: Vec::new(),
        eval_arc: generator.eval_batches_for::<B>(
            TaskFamily::ArcGrid,
            config.eval_batch_size,
            config.effective_arc_eval_batches(),
            device,
        )?,
    })
}

#[derive(Clone, Debug)]
struct SplitBatches<B: AutodiffBackend> {
    batches: Vec<TokenBatch<B>>,
    model_facing_documents: usize,
    bridge_documents: usize,
    bridge_chunks: usize,
    bridge_tokens: usize,
    native_documents: usize,
    native_chunks: usize,
    native_tokens: usize,
    sequence_len: usize,
}

fn build_split_batches<B, N>(
    tokenizer: &FaceoffTokenizer,
    vocab: &fractal_tokenizer::FaceoffVocab,
    documents: &[String],
    batch_size: usize,
    training_input: &TrainingInputSpec,
    native_tokenizer: &N,
    chunk_limits: &FaceoffChunkLimits,
    truncation_policy: &NativeTruncationPolicy,
    device: &B::Device,
) -> Result<SplitBatches<B>, FractalError>
where
    B: AutodiffBackend,
    N: NativeTokenizer<Token = u32>,
    N::Error: fmt::Display + fmt::Debug + Send + Sync + 'static,
{
    let model_facing_documents = documents
        .iter()
        .map(|text| {
            let encoded = tokenizer
                .encode_text_v2::<B>(text, vocab, device)
                .map_err(|error| FractalError::InvalidState(error.to_string()))?;
            ModelFacingDocument::from_encoded(encoded, *chunk_limits)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let model_facing_batch = ModelFacingBatch::from(model_facing_documents);
    let bridge_batch = TypedEmbeddingBridge
        .bridge_batch(&model_facing_batch)
        .map_err(|error| FractalError::InvalidState(error.to_string()))?;
    let native_batch = NativeCompatibilityAdapter
        .retokenize_batch(&model_facing_batch, native_tokenizer)
        .map_err(native_error_to_state)?;
    let native_collated = native_batch
        .collate(
            &NativeCollationSpec::new(
                training_input
                    .tokenizer
                    .as_ref()
                    .ok_or_else(|| {
                        FractalError::InvalidState(
                            "tokenizer metadata was unexpectedly missing".into(),
                        )
                    })?
                    .pad_token_id as u32,
            )
            .with_max_sequence_len(
                NonZeroUsize::new(chunk_limits.max_tokens_per_chunk.max(1)).unwrap(),
            )
            .with_truncation_policy(*truncation_policy),
        )
        .map_err(|error| FractalError::InvalidState(error.to_string()))?;

    let sequences = native_collated
        .chunks()
        .map(sequence_from_native_chunk)
        .collect::<Result<Vec<_>, _>>()?;
    let batches = sequences_into_batches::<B>(
        sequences,
        batch_size,
        TaskFamily::TokenizerBackedText,
        device,
    )?;

    let native_documents = native_collated.len();
    let native_chunks = native_collated.chunk_count();
    let native_tokens = native_collated
        .chunks()
        .map(|chunk| chunk.valid_token_count())
        .sum();
    let sequence_len = native_collated.sequence_len;

    Ok(SplitBatches {
        batches,
        model_facing_documents: model_facing_batch.len(),
        bridge_documents: bridge_batch.len(),
        bridge_chunks: bridge_batch.chunk_count(),
        bridge_tokens: bridge_batch.token_count(),
        native_documents,
        native_chunks,
        native_tokens,
        sequence_len,
    })
}

#[derive(Clone, Debug)]
struct SequenceExample {
    input: Vec<i64>,
    target: Vec<i64>,
}

fn sequence_from_native_chunk(
    chunk: &fractal_tokenizer::NativeCollatedChunk<u32>,
) -> Result<SequenceExample, FractalError> {
    let valid_len = chunk.valid_token_count();
    let input = chunk
        .padded_tokens
        .iter()
        .map(|token| *token as i64)
        .collect::<Vec<_>>();
    if input.is_empty() {
        return Err(FractalError::InvalidState(
            "tokenizer-backed chunk produced an empty token sequence".into(),
        ));
    }
    let mut target = input.clone();
    target.rotate_left(1);
    if let Some(last) = target.last_mut() {
        *last = PAD_TOKEN as i64;
    }

    if valid_len == 0 {
        return Err(FractalError::InvalidState(
            "tokenizer-backed chunk produced zero valid tokens".into(),
        ));
    }

    Ok(SequenceExample { input, target })
}

fn sequences_into_batches<B: Backend>(
    sequences: Vec<SequenceExample>,
    batch_size: usize,
    family: TaskFamily,
    device: &B::Device,
) -> Result<Vec<TokenBatch<B>>, FractalError> {
    if sequences.is_empty() {
        return Err(FractalError::InvalidConfig(
            "tokenizer-backed text split must produce at least one sequence".into(),
        ));
    }
    if batch_size == 0 {
        return Err(FractalError::InvalidConfig(
            "tokenizer-backed batch size must be greater than zero".into(),
        ));
    }

    let seq_len = sequences
        .first()
        .map(|sequence| sequence.input.len())
        .ok_or_else(|| {
            FractalError::InvalidState("tokenizer-backed sequence set was empty".into())
        })?;
    if sequences
        .iter()
        .any(|sequence| sequence.input.len() != seq_len || sequence.target.len() != seq_len)
    {
        return Err(FractalError::InvalidState(
            "tokenizer-backed sequences must share a common sequence length".into(),
        ));
    }

    let mut batches = Vec::new();
    for chunk in sequences.chunks(batch_size) {
        let mut input_flat = Vec::with_capacity(chunk.len() * seq_len);
        let mut target_flat = Vec::with_capacity(chunk.len() * seq_len);
        for sequence in chunk {
            input_flat.extend_from_slice(&sequence.input);
            target_flat.extend_from_slice(&sequence.target);
        }
        let input_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(input_flat, [chunk.len(), seq_len]),
            device,
        );
        let target_ids = Tensor::<B, 2, Int>::from_data(
            TensorData::new(target_flat, [chunk.len(), seq_len]),
            device,
        );
        batches.push(TokenBatch {
            input_ids,
            target_ids,
            family,
        });
    }

    Ok(batches)
}

fn native_error_to_state<E: fmt::Display>(error: NativeCompatibilityError<E>) -> FractalError {
    FractalError::InvalidState(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, env, error::Error, fmt, fs};

    use burn::backend::candle::CandleDevice;
    use fractal_core::lifecycle::ArcSourceMode;
    use fractal_core::{
        ArtifactPolicy, BudgetSpec, ComparisonContract, DecisionIntent, ExecutionBackend,
        ExecutionTarget, ExecutionTargetKind, ExperimentId, ExperimentQuestion,
        ExperimentSpecTemplate, LaneIntent, PrimitiveVariantName, RuntimeSurfaceSpec, SpeciesId,
        TaskFamily, TrainingInputMode, TrainingInputSpec,
    };

    use crate::{
        aggregate_results, persist_run_artifacts, species_registry_for_species, TournamentLane,
        TournamentPreset, TournamentRunReport,
    };

    use super::{
        build_tokenizer_backed_batches, run_tokenizer_backed_species, TokenizerArtifactSpec,
        TokenizerTrainingCorpus, TokenizerTrainingRuntime,
    };
    use fractal_core::registry::take_last_species_run_artifact;
    use fractal_tokenizer::NativeTokenizer;

    #[derive(Clone, Debug)]
    struct ByteTokenizer;

    #[derive(Clone, Debug)]
    struct ByteTokenizerError;

    impl fmt::Display for ByteTokenizerError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("byte tokenizer error")
        }
    }

    impl Error for ByteTokenizerError {}

    impl NativeTokenizer for ByteTokenizer {
        type Token = u32;
        type Error = ByteTokenizerError;

        fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
            Ok(text
                .as_bytes()
                .iter()
                .map(|byte| (*byte as u32) + 1)
                .collect())
        }
    }

    #[test]
    fn tokenizer_backed_stage0_smoke_path_captures_bridge_metadata() {
        let temp_root =
            env::temp_dir().join(format!("fractal-tokenizer-stage0-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        let artifact_dir = temp_root.join("artifacts");
        let manifest_dir = temp_root.join("manifests");
        env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let mut config = TournamentPreset::FastTest.config();
        config.dim = 8;
        config.levels = 2;
        config.vocab_size = 32_000;
        config.max_seq_len = 64;
        config.max_recursion_depth = 2;
        config.stability_depth = 1;
        config.train_batch_size = 1;
        config.eval_batch_size = 1;
        config.train_steps_per_species = 1;
        config.eval_batches_per_family = 1;
        config.perplexity_eval_batches = Some(1);
        config.arc_eval_batches = Some(1);
        config.learning_rate = 1e-3;

        let training_input = TrainingInputSpec::tokenizer_backed_text(
            "fineweb-stage0-smoke",
            TokenizerArtifactSpec {
                artifact_id: "frozen-32k-sentencepiece".to_owned(),
                artifact_path: Some("inline://frozen-32k-sentencepiece".to_owned()),
                vocab_size: 32_000,
                pad_token_id: 0,
            },
        );
        let template = ExperimentSpecTemplate {
            experiment_id: ExperimentId {
                logical_name: "stage0-tokenizer-smoke".to_owned(),
                run_id: "run-123".to_owned(),
                branch: Some("codex/stage0-tokenizer-training-path".to_owned()),
                commit_sha: Some("abc123".to_owned()),
                created_at_unix_ms: 123,
            },
            question: ExperimentQuestion {
                summary: "exercise tokenizer-backed Stage 0 text training".to_owned(),
                lane_intent: LaneIntent::Winner,
                decision_intent: DecisionIntent::Benchmark,
            },
            budget: BudgetSpec::from_config(TournamentPreset::FastTest, &config),
            training_input: training_input.clone(),
            runtime: RuntimeSurfaceSpec::default(),
            comparison: ComparisonContract::authoritative_same_preset(),
            execution: ExecutionTarget {
                kind: ExecutionTargetKind::Local,
                backend: ExecutionBackend::from_compute_backend(&config.execution_backend),
                execution_mode: config.execution_mode,
                pod_id: None,
                wrapper_timeout_seconds: None,
            },
            artifacts: ArtifactPolicy::default(),
        };
        let experiment = template.resolve_variant(
            SpeciesId::P1Contractive,
            PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
        );
        let corpus = TokenizerTrainingCorpus::new(
            "fineweb-stage0-smoke",
            vec![
                "the quick brown fox jumps over the lazy dog".to_owned(),
                "recursive primitives keep their own state".to_owned(),
            ],
            vec!["held out text stays separate".to_owned()],
        );
        let runtime = TokenizerTrainingRuntime::new(ByteTokenizer, config.max_seq_len).unwrap();
        let device = CandleDevice::Cpu;

        let (metrics, bridge_stats) =
            run_tokenizer_backed_species::<fractal_core::CpuTrainBackend, _>(
                SpeciesId::P1Contractive,
                PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
                config.clone(),
                Some(experiment.clone()),
                &corpus,
                &runtime,
                device,
            )
            .expect("tokenizer-backed stage0 smoke run succeeds");

        assert!(metrics.tokens_per_sec.is_finite());
        assert!(bridge_stats.bridge_enabled);
        assert!(bridge_stats.bridge_observational_only);
        assert_eq!(
            bridge_stats.training_input_mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(bridge_stats.corpus_name, "fineweb-stage0-smoke");
        assert_eq!(
            bridge_stats.tokenizer_artifact_id,
            "frozen-32k-sentencepiece"
        );
        assert_eq!(
            bridge_stats.arc_source_mode,
            ArcSourceMode::SyntheticCanonical
        );
        let artifact = take_last_species_run_artifact()
            .expect("tokenizer-backed stage0 smoke run records an artifact");
        assert_eq!(
            artifact
                .manifest
                .experiment
                .as_ref()
                .expect("experiment recorded")
                .training_input
                .mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(
            artifact
                .manifest
                .experiment
                .as_ref()
                .expect("experiment recorded")
                .training_input
                .arc_source
                .mode,
            ArcSourceMode::SyntheticCanonical
        );

        let report = TournamentRunReport::new(
            TournamentPreset::FastTest,
            TournamentLane::Leader,
            ComparisonContract::authoritative_same_preset(),
            config,
            species_registry_for_species(SpeciesId::P1Contractive),
            aggregate_results(vec![metrics.clone()]),
            fractal_core::TournamentRunArtifact {
                config: TournamentPreset::FastTest.config(),
                species: vec![artifact],
            },
            BTreeMap::from([(SpeciesId::P1Contractive, bridge_stats.clone())]),
        );

        let paths = persist_run_artifacts(&report).expect("tokenizer-backed report persists");
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.manifest_path).expect("manifest readable"))
                .expect("manifest json parses");
        let artifact_json: serde_json::Value =
            serde_json::from_slice(&fs::read(&paths.artifact_path).expect("artifact readable"))
                .expect("artifact json parses");

        assert_eq!(
            manifest["experiments"][0]["training_input"]["mode"],
            serde_json::Value::String("tokenizer-backed-text".to_owned())
        );
        assert_eq!(
            manifest["experiments"][0]["training_input"]["tokenizer"]["artifact_id"],
            serde_json::Value::String("frozen-32k-sentencepiece".to_owned())
        );
        assert_eq!(
            artifact_json["results"][0]["tokenizer_bridge"]["bridge_enabled"],
            serde_json::Value::Bool(true)
        );
        assert_eq!(
            artifact_json["results"][0]["tokenizer_bridge"]["training_input_mode"],
            serde_json::Value::String("tokenizer-backed-text".to_owned())
        );

        env::remove_var("FRACTAL_RUN_ARTIFACT_DIR");
        env::remove_var("FRACTAL_RUN_MANIFEST_DIR");
        let _ = fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn tokenizer_bridge_batch_builds_from_inline_text() {
        let device = CandleDevice::Cpu;
        let mut config = TournamentPreset::FastTest.config();
        config.dim = 8;
        config.levels = 2;
        config.vocab_size = 32_000;
        config.max_seq_len = 16;
        config.max_recursion_depth = 2;
        config.stability_depth = 1;
        config.train_batch_size = 1;
        config.eval_batch_size = 1;
        let training_input = TrainingInputSpec::tokenizer_backed_text(
            "fineweb-stage0-smoke",
            TokenizerArtifactSpec {
                artifact_id: "frozen-32k-sentencepiece".to_owned(),
                artifact_path: Some("inline://frozen-32k-sentencepiece".to_owned()),
                vocab_size: 32_000,
                pad_token_id: 0,
            },
        );
        let corpus = TokenizerTrainingCorpus::new(
            "fineweb-stage0-smoke",
            vec!["alpha beta".to_owned()],
            vec!["gamma delta".to_owned()],
        );
        let runtime = TokenizerTrainingRuntime::new(ByteTokenizer, config.max_seq_len).unwrap();
        let (batches, stats) = build_tokenizer_backed_batches::<fractal_core::CpuTrainBackend, _>(
            &config,
            &training_input,
            &corpus,
            &runtime,
            &device,
        )
        .expect("tokenizer-backed batches build");

        assert_eq!(batches.train_sentence.len(), 1);
        assert_eq!(batches.eval_sentence.len(), 1);
        assert_eq!(
            batches.train_sentence[0].family,
            TaskFamily::TokenizerBackedText
        );
        assert_eq!(
            batches.eval_sentence[0].family,
            TaskFamily::TokenizerBackedText
        );
        assert_eq!(batches.train_arc[0].family, TaskFamily::ArcGrid);
        assert_eq!(batches.eval_arc[0].family, TaskFamily::ArcGrid);
        assert_eq!(
            stats.training_input_mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(stats.arc_source_mode, ArcSourceMode::SyntheticCanonical);
        assert!(stats.bridge_tokens > 0);
        assert!(stats.native_tokens > 0);
    }
}
