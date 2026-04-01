use std::{
    fmt, fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::Arc,
};

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
use sentencepiece::{SentencePieceError, SentencePieceProcessor};

pub const STAGE0_CANONICAL_TOKENIZER_REPO_ID: &str = "openlm-research/open_llama_3b_v2";
pub const STAGE0_CANONICAL_TOKENIZER_FILENAME: &str = "tokenizer.model";
pub const STAGE0_CANONICAL_TOKENIZER_USE_FAST: bool = false;

#[derive(Clone, Debug)]
pub struct Stage0SlowTokenizer {
    processor: Arc<SentencePieceProcessor>,
}

impl Stage0SlowTokenizer {
    pub fn open(path: &Path) -> Result<Self, Stage0SlowTokenizerError> {
        let processor = SentencePieceProcessor::open(path).map_err(|source| {
            Stage0SlowTokenizerError::Load {
                path: path.to_path_buf(),
                reason: source,
            }
        })?;
        Ok(Self {
            processor: Arc::new(processor),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.processor.len()
    }

    pub fn model_pad_token_id(&self) -> Option<u32> {
        self.processor.pad_id()
    }
}

impl NativeTokenizer for Stage0SlowTokenizer {
    type Token = u32;
    type Error = Stage0SlowTokenizerError;

    fn tokenize(&self, text: &str) -> Result<Vec<Self::Token>, Self::Error> {
        self.processor
            .encode(text)
            .map(|pieces| pieces.into_iter().map(|piece| piece.id).collect())
            .map_err(|source| Stage0SlowTokenizerError::Encode {
                input_preview: text.chars().take(32).collect(),
                reason: source,
            })
    }
}

#[derive(Debug)]
pub enum Stage0SlowTokenizerError {
    Load {
        path: PathBuf,
        reason: SentencePieceError,
    },
    Encode {
        input_preview: String,
        reason: SentencePieceError,
    },
}

impl fmt::Display for Stage0SlowTokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load { path, reason } => {
                write!(
                    f,
                    "failed to load slow Stage 0 tokenizer from {}: {reason}",
                    path.display()
                )
            }
            Self::Encode {
                input_preview,
                reason,
            } => write!(
                f,
                "failed to encode text with slow Stage 0 tokenizer for input {:?}: {reason}",
                input_preview
            ),
        }
    }
}

impl std::error::Error for Stage0SlowTokenizerError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextCorpusFormat {
    JsonlText { text_field: String },
    PlainTextLines,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextCorpusSplitSource {
    pub path: PathBuf,
    pub format: TextCorpusFormat,
    pub max_documents: Option<usize>,
}

impl TextCorpusSplitSource {
    pub fn jsonl_text(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: TextCorpusFormat::JsonlText {
                text_field: "text".to_owned(),
            },
            max_documents: None,
        }
    }

    pub fn plain_text_lines(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            format: TextCorpusFormat::PlainTextLines,
            max_documents: None,
        }
    }

    pub fn with_max_documents(mut self, max_documents: usize) -> Self {
        self.max_documents = Some(max_documents);
        self
    }

    fn load_documents(&self) -> Result<Vec<String>, FractalError> {
        let content = fs::read_to_string(&self.path).map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to read text corpus split {}: {error}",
                self.path.display()
            ))
        })?;
        let mut documents = match &self.format {
            TextCorpusFormat::JsonlText { text_field } => content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(|line| {
                    let value: serde_json::Value = serde_json::from_str(line).map_err(|error| {
                        FractalError::InvalidState(format!(
                            "failed to parse jsonl line for {}: {error}",
                            self.path.display()
                        ))
                    })?;
                    let text = value
                        .get(text_field)
                        .and_then(|value| value.as_str())
                        .ok_or_else(|| {
                            FractalError::InvalidState(format!(
                                "jsonl record in {} is missing string field {}",
                                self.path.display(),
                                text_field
                            ))
                        })?;
                    Ok(text.to_owned())
                })
                .collect::<Result<Vec<_>, FractalError>>()?,
            TextCorpusFormat::PlainTextLines => content
                .lines()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>(),
        };

        if let Some(max_documents) = self.max_documents {
            documents.truncate(max_documents);
        }
        Ok(documents)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerTrainingCorpusSource {
    pub name: String,
    pub train: TextCorpusSplitSource,
    pub eval: TextCorpusSplitSource,
}

impl TokenizerTrainingCorpusSource {
    pub fn new(
        name: impl Into<String>,
        train: TextCorpusSplitSource,
        eval: TextCorpusSplitSource,
    ) -> Self {
        Self {
            name: name.into(),
            train,
            eval,
        }
    }

    pub fn fineweb_jsonl(
        name: impl Into<String>,
        train_path: impl Into<PathBuf>,
        eval_path: impl Into<PathBuf>,
    ) -> Self {
        Self::new(
            name,
            TextCorpusSplitSource::jsonl_text(train_path),
            TextCorpusSplitSource::jsonl_text(eval_path),
        )
    }

    pub fn load(&self) -> Result<TokenizerTrainingCorpus, FractalError> {
        let corpus = TokenizerTrainingCorpus::new(
            self.name.clone(),
            self.train.load_documents()?,
            self.eval.load_documents()?,
        );
        corpus.validate()?;
        Ok(corpus)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedTokenizerArtifact {
    pub repo_id: String,
    pub revision: Option<String>,
    pub tokenizer_filename: String,
    pub use_fast: bool,
    pub local_path: PathBuf,
    pub spec: TokenizerArtifactSpec,
}

impl ResolvedTokenizerArtifact {
    pub fn from_training_input(training_input: &TrainingInputSpec) -> Result<Self, FractalError> {
        let spec = training_input.tokenizer.clone().ok_or_else(|| {
            FractalError::InvalidConfig(
                "tokenizer-backed text training requires tokenizer artifact metadata".into(),
            )
        })?;
        let path = spec.artifact_path.clone().ok_or_else(|| {
            FractalError::InvalidConfig(
                "tokenizer-backed text training requires a tokenizer artifact path".into(),
            )
        })?;
        let local_path = PathBuf::from(path);
        if !local_path.is_file() {
            return Err(FractalError::InvalidState(format!(
                "tokenizer artifact path {} does not point to a readable file",
                local_path.display()
            )));
        }
        let tokenizer_filename = local_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(STAGE0_CANONICAL_TOKENIZER_FILENAME)
            .to_owned();
        Ok(Self {
            repo_id: spec.artifact_id.clone(),
            revision: None,
            tokenizer_filename,
            use_fast: STAGE0_CANONICAL_TOKENIZER_USE_FAST,
            local_path,
            spec,
        })
    }

    pub fn canonical_open_llama_3b_v2(
        tokenizer_model_path: impl Into<PathBuf>,
        vocab_size: usize,
        pad_token_id: usize,
    ) -> Self {
        let local_path = tokenizer_model_path.into();
        let tokenizer_filename = local_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(STAGE0_CANONICAL_TOKENIZER_FILENAME)
            .to_owned();
        let spec = TokenizerArtifactSpec {
            artifact_id: STAGE0_CANONICAL_TOKENIZER_REPO_ID.to_owned(),
            artifact_path: Some(local_path.display().to_string()),
            vocab_size,
            pad_token_id,
        };
        Self {
            repo_id: STAGE0_CANONICAL_TOKENIZER_REPO_ID.to_owned(),
            revision: None,
            tokenizer_filename,
            use_fast: STAGE0_CANONICAL_TOKENIZER_USE_FAST,
            local_path,
            spec,
        }
    }

    pub fn into_training_input(self, corpus_name: impl Into<String>) -> TrainingInputSpec {
        TrainingInputSpec::tokenizer_backed_text(corpus_name, self.spec)
    }
}

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

fn validate_stage0_slow_tokenizer_artifact(
    artifact: &ResolvedTokenizerArtifact,
) -> Result<(), FractalError> {
    if artifact.repo_id != STAGE0_CANONICAL_TOKENIZER_REPO_ID {
        return Err(FractalError::InvalidConfig(format!(
            "Stage 0 canonical tokenizer must use artifact {}, got {}",
            STAGE0_CANONICAL_TOKENIZER_REPO_ID, artifact.repo_id
        )));
    }
    if artifact.use_fast {
        return Err(FractalError::InvalidConfig(
            "Stage 0 canonical tokenizer must use slow-tokenizer semantics".into(),
        ));
    }
    if artifact.tokenizer_filename != STAGE0_CANONICAL_TOKENIZER_FILENAME {
        return Err(FractalError::InvalidConfig(format!(
            "Stage 0 canonical tokenizer must resolve {} but found {}",
            STAGE0_CANONICAL_TOKENIZER_FILENAME, artifact.tokenizer_filename
        )));
    }
    Ok(())
}

pub fn load_stage0_slow_tokenizer(
    artifact: &ResolvedTokenizerArtifact,
) -> Result<Stage0SlowTokenizer, FractalError> {
    validate_stage0_slow_tokenizer_artifact(artifact)?;
    let tokenizer = Stage0SlowTokenizer::open(Path::new(&artifact.local_path))
        .map_err(|error| FractalError::InvalidState(error.to_string()))?;

    if tokenizer.vocab_size() != artifact.spec.vocab_size {
        return Err(FractalError::InvalidConfig(format!(
            "tokenizer artifact {} reports vocab size {} but slow tokenizer model provides {}",
            artifact.local_path.display(),
            artifact.spec.vocab_size,
            tokenizer.vocab_size()
        )));
    }

    if let Some(model_pad_id) = tokenizer.model_pad_token_id() {
        if model_pad_id as usize != artifact.spec.pad_token_id {
            return Err(FractalError::InvalidConfig(format!(
                "tokenizer artifact {} reports pad token id {} but slow tokenizer model provides {}",
                artifact.local_path.display(),
                artifact.spec.pad_token_id,
                model_pad_id
            )));
        }
    }

    Ok(tokenizer)
}

pub fn load_stage0_tokenizer_runtime(
    training_input: &TrainingInputSpec,
    max_sequence_len: usize,
) -> Result<
    (
        TokenizerTrainingRuntime<Stage0SlowTokenizer>,
        ResolvedTokenizerArtifact,
    ),
    FractalError,
> {
    let artifact = ResolvedTokenizerArtifact::from_training_input(training_input)?;
    let tokenizer = load_stage0_slow_tokenizer(&artifact)?;
    let runtime = TokenizerTrainingRuntime::new(tokenizer, max_sequence_len)?;
    Ok((runtime, artifact))
}

pub fn build_tokenizer_backed_batches_from_source<B>(
    config: &fractal_core::TournamentConfig,
    training_input: &TrainingInputSpec,
    corpus_source: &TokenizerTrainingCorpusSource,
    device: &B::Device,
) -> Result<
    (
        TrainingBatchSet<B>,
        TokenizerBridgeStats,
        ResolvedTokenizerArtifact,
    ),
    FractalError,
>
where
    B: AutodiffBackend,
{
    let corpus = corpus_source.load()?;
    let (runtime, artifact) = load_stage0_tokenizer_runtime(training_input, config.max_seq_len)?;
    let (batches, stats) =
        build_tokenizer_backed_batches::<B, _>(config, training_input, &corpus, &runtime, device)?;
    Ok((batches, stats, artifact))
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

    let arc_eval_batches = build_canonical_arc_eval_batches::<B>(config, device)?;

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
            train_arc: None,
            eval_sentence: eval.batches.clone(),
            eval_arc: arc_eval_batches,
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

pub fn run_tokenizer_backed_species_from_source<B>(
    species: SpeciesId,
    variant_name: PrimitiveVariantName,
    config: fractal_core::TournamentConfig,
    experiment: Option<ExperimentSpec>,
    corpus_source: &TokenizerTrainingCorpusSource,
    device: B::Device,
) -> Result<
    (
        SpeciesRawMetrics,
        TokenizerBridgeStats,
        ResolvedTokenizerArtifact,
    ),
    FractalError,
>
where
    B: AutodiffBackend,
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
    let (batches, stats, artifact) = build_tokenizer_backed_batches_from_source::<B>(
        &config,
        training_input,
        corpus_source,
        &device,
    )?;

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

    Ok((metrics, stats, artifact))
}

fn build_canonical_arc_eval_batches<B: AutodiffBackend>(
    config: &fractal_core::TournamentConfig,
    device: &B::Device,
) -> Result<Vec<TokenBatch<B>>, FractalError> {
    let generator = SimpleHierarchicalGenerator::new(GeneratorConfig {
        vocab_size: config.vocab_size,
        max_seq_len: config.max_seq_len,
        train_examples_per_family: config.train_batch_size.max(1),
        eval_examples_per_family: config.eval_batches_per_family.max(1),
        seed: config.seed,
        depth_config: config.generator_depth_config,
    })?;

    generator.eval_batches_for::<B>(
        TaskFamily::ArcGrid,
        config.eval_batch_size,
        config.effective_arc_eval_batches(),
        device,
    )
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
    valid_token_count: usize,
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

    Ok(SequenceExample {
        input,
        target,
        valid_token_count: valid_len,
    })
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
        let mut token_count = 0usize;
        for sequence in chunk {
            input_flat.extend_from_slice(&sequence.input);
            target_flat.extend_from_slice(&sequence.target);
            token_count += sequence.valid_token_count;
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
            token_count,
        });
    }

    Ok(batches)
}

fn native_error_to_state<E: fmt::Display>(error: NativeCompatibilityError<E>) -> FractalError {
    FractalError::InvalidState(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        env,
        error::Error,
        fmt, fs,
        path::PathBuf,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::backend::candle::CandleDevice;
    use fractal_core::lifecycle::ArcSourceMode;
    use fractal_core::{
        ArtifactPolicy, BudgetSpec, ComparisonContract, DecisionIntent, ExecutionBackend,
        ExecutionTarget, ExecutionTargetKind, ExperimentId, ExperimentQuestion,
        ExperimentSpecTemplate, LaneIntent, OptimizerSpec, PrimitiveVariantName,
        RuntimeSurfaceSpec, SpeciesId, TaskFamily, TrainingInputMode, TrainingInputSpec,
    };

    use crate::{
        aggregate_results, persist_run_artifacts, species_registry_for_species, TournamentLane,
        TournamentPreset, TournamentRunReport,
    };

    use super::{
        build_tokenizer_backed_batches, load_stage0_tokenizer_runtime,
        run_tokenizer_backed_species_from_source, ResolvedTokenizerArtifact, TokenizerArtifactSpec,
        TokenizerTrainingCorpus, TokenizerTrainingCorpusSource, TokenizerTrainingRuntime,
        STAGE0_CANONICAL_TOKENIZER_FILENAME, STAGE0_CANONICAL_TOKENIZER_REPO_ID,
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

    fn unique_temp_path(prefix: &str, suffix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "fractal-stage0-{prefix}-{}-{stamp}{suffix}",
            std::process::id()
        ))
    }

    fn write_jsonl_corpus(path: &PathBuf, documents: &[&str]) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let body = documents
            .iter()
            .map(|document| serde_json::json!({ "text": document }).to_string())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, format!("{body}\n")).unwrap();
    }

    fn sentencepiece_testdata_model() -> PathBuf {
        let output = Command::new("cargo")
            .args(["metadata", "--format-version", "1"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("cargo metadata should run");
        assert!(
            output.status.success(),
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let metadata: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("cargo metadata should be valid json");
        let packages = metadata["packages"]
            .as_array()
            .expect("cargo metadata should include packages");
        let manifest_path = packages
            .iter()
            .find(|package| package["name"] == "sentencepiece")
            .and_then(|package| package["manifest_path"].as_str())
            .expect("sentencepiece package should be present in cargo metadata");
        PathBuf::from(manifest_path)
            .parent()
            .expect("sentencepiece manifest should have parent")
            .join("testdata")
            .join("toy.model")
    }

    fn build_file_backed_sentencepiece_tokenizer() -> PathBuf {
        let dir = unique_temp_path("tokenizer-artifact", "");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join(STAGE0_CANONICAL_TOKENIZER_FILENAME);
        fs::copy(sentencepiece_testdata_model(), &path).unwrap();
        path
    }

    #[test]
    fn tokenizer_backed_stage0_smoke_path_captures_bridge_metadata() {
        let temp_root =
            env::temp_dir().join(format!("fractal-tokenizer-stage0-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_root);
        let artifact_dir = temp_root.join("artifacts");
        let manifest_dir = temp_root.join("manifests");
        let train_path = temp_root.join("train.jsonl");
        let eval_path = temp_root.join("eval.jsonl");
        env::set_var("FRACTAL_RUN_ARTIFACT_DIR", &artifact_dir);
        env::set_var("FRACTAL_RUN_MANIFEST_DIR", &manifest_dir);

        let train_documents = [
            "I saw a girl with a telescope.",
            "I saw a girl with a telescope.",
        ];
        let eval_documents = ["I saw a girl with a telescope."];
        write_jsonl_corpus(&train_path, &train_documents);
        write_jsonl_corpus(&eval_path, &eval_documents);
        let tokenizer_path = build_file_backed_sentencepiece_tokenizer();
        let resolved_tokenizer =
            ResolvedTokenizerArtifact::canonical_open_llama_3b_v2(&tokenizer_path, 1_000, 0);

        let mut config = TournamentPreset::FastTest.config();
        config.dim = 8;
        config.levels = 2;
        config.vocab_size = 1_000;
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
        config.optimizer = OptimizerSpec::legacy_adam(config.learning_rate);

        let training_input = resolved_tokenizer
            .clone()
            .into_training_input("fineweb-stage0-smoke");
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
            optimizer: config.optimizer.clone(),
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
        let corpus_source = TokenizerTrainingCorpusSource::fineweb_jsonl(
            "fineweb-stage0-smoke",
            &train_path,
            &eval_path,
        );
        let (_runtime, loaded_artifact) =
            load_stage0_tokenizer_runtime(&training_input, config.max_seq_len)
                .expect("canonical Stage 0 tokenizer runtime loads");
        assert_eq!(loaded_artifact.repo_id, STAGE0_CANONICAL_TOKENIZER_REPO_ID);
        assert_eq!(loaded_artifact.local_path, tokenizer_path);
        assert_eq!(
            loaded_artifact.tokenizer_filename,
            STAGE0_CANONICAL_TOKENIZER_FILENAME
        );
        assert!(!loaded_artifact.use_fast);
        let device = CandleDevice::Cpu;

        let (metrics, bridge_stats, source_artifact) =
            run_tokenizer_backed_species_from_source::<fractal_core::CpuTrainBackend>(
                SpeciesId::P1Contractive,
                PrimitiveVariantName::new_unchecked("p1_contractive_v1"),
                config.clone(),
                Some(experiment.clone()),
                &corpus_source,
                device,
            )
            .expect("tokenizer-backed stage0 smoke run succeeds");

        assert!(metrics.tokens_per_sec.is_finite());
        assert_eq!(source_artifact.repo_id, STAGE0_CANONICAL_TOKENIZER_REPO_ID);
        assert!(bridge_stats.bridge_enabled);
        assert!(bridge_stats.bridge_observational_only);
        assert_eq!(
            bridge_stats.training_input_mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(bridge_stats.corpus_name, "fineweb-stage0-smoke");
        assert_eq!(
            bridge_stats.tokenizer_artifact_id,
            STAGE0_CANONICAL_TOKENIZER_REPO_ID
        );
        assert_eq!(
            bridge_stats.arc_source_mode,
            ArcSourceMode::SyntheticCanonical
        );
        assert!(bridge_stats.bridge_enabled);
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
        assert!(artifact
            .manifest
            .experiment
            .as_ref()
            .expect("experiment recorded")
            .training_input
            .tokenizer
            .is_some());

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
            serde_json::Value::String(STAGE0_CANONICAL_TOKENIZER_REPO_ID.to_owned())
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
        if let Some(parent) = tokenizer_path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
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
        assert!(batches.train_arc.is_none());
        assert_eq!(batches.eval_arc[0].family, TaskFamily::ArcGrid);
        assert_eq!(
            stats.training_input_mode,
            TrainingInputMode::TokenizerBackedText
        );
        assert_eq!(stats.arc_source_mode, ArcSourceMode::SyntheticCanonical);
        assert!(stats.bridge_tokens > 0);
        assert!(stats.native_tokens > 0);
    }

    #[test]
    fn stage0_runtime_uses_real_slow_sentencepiece_tokens() {
        let tokenizer_path = build_file_backed_sentencepiece_tokenizer();
        let training_input =
            ResolvedTokenizerArtifact::canonical_open_llama_3b_v2(&tokenizer_path, 1_000, 0)
                .into_training_input("fineweb-stage0-smoke");

        let (runtime, artifact) = load_stage0_tokenizer_runtime(&training_input, 64)
            .expect("slow sentencepiece stage0 runtime should load");
        let tokens = runtime
            .tokenizer
            .tokenize("I saw a girl with a telescope.")
            .expect("slow tokenizer should encode canonical sentence");

        assert_eq!(artifact.repo_id, STAGE0_CANONICAL_TOKENIZER_REPO_ID);
        assert_eq!(
            artifact.tokenizer_filename,
            STAGE0_CANONICAL_TOKENIZER_FILENAME
        );
        assert_eq!(
            tokens,
            vec![8, 465, 10, 947, 41, 10, 170, 168, 110, 28, 20, 143, 4]
        );

        if let Some(parent) = tokenizer_path.parent() {
            let _ = fs::remove_dir_all(parent);
        }
    }

    #[test]
    fn stage0_runtime_rejects_fast_tokenizer_artifact_filename() {
        let dir = unique_temp_path("tokenizer-artifact-invalid", "");
        fs::create_dir_all(&dir).unwrap();
        let invalid_path = dir.join("tokenizer.json");
        fs::copy(sentencepiece_testdata_model(), &invalid_path).unwrap();

        let training_input =
            ResolvedTokenizerArtifact::canonical_open_llama_3b_v2(&invalid_path, 1_000, 0)
                .into_training_input("fineweb-stage0-smoke");
        let error = load_stage0_tokenizer_runtime(&training_input, 64)
            .expect_err("stage0 runtime must reject fast-tokenizer artifact filenames");

        assert!(
            error
                .to_string()
                .contains(STAGE0_CANONICAL_TOKENIZER_FILENAME),
            "unexpected error: {error}"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
