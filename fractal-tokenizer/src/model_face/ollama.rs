use std::time::{Duration, Instant};

use fractal_core::error::FractalError;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

use crate::{
    build_recursive_overlay, HuggingFaceNativeTokenizer, OverlayModelFacingBatch,
    OverlayModelFacingDocument, OverlayTransportAdapter, OverlayTransportConfig,
    RecursiveOverlayConfig, RecursiveOverlayMode,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OllamaEndpointConfig {
    pub base_url: String,
    pub embedding_model: String,
}

impl Default for OllamaEndpointConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:11434".to_string(),
            embedding_model: "nomic-embed-text:latest".to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OllamaClientBenchmarkConfig {
    pub warmup_rounds: usize,
    pub measure_rounds: usize,
}

impl Default for OllamaClientBenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_rounds: 2,
            measure_rounds: 5,
        }
    }
}

#[derive(Clone)]
pub struct OverlayBenchmarkRequest<'a> {
    pub tokenizer: &'a HuggingFaceNativeTokenizer,
    pub texts: &'a [String],
    pub overlay_mode: RecursiveOverlayMode,
    pub overlay_config: &'a RecursiveOverlayConfig,
    pub transport_config: OverlayTransportConfig,
}

#[derive(Clone, Debug)]
pub struct OllamaEmbeddingClient {
    client: Client,
    config: OllamaEndpointConfig,
}

impl OllamaEmbeddingClient {
    pub fn new(config: OllamaEndpointConfig) -> Result<Self, FractalError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to build Ollama HTTP client for {}: {error}",
                    config.base_url
                ))
            })?;
        Ok(Self { client, config })
    }

    pub fn config(&self) -> &OllamaEndpointConfig {
        &self.config
    }

    pub fn available_models(&self) -> Result<Vec<String>, FractalError> {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        let response = self
            .client
            .get(&url)
            .send()
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to query Ollama tags from {url}: {error}"
                ))
            })?
            .error_for_status()
            .map_err(|error| {
                FractalError::InvalidState(format!("Ollama tags request failed for {url}: {error}"))
            })?;
        let payload: OllamaTagsResponse = response.json().map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to decode Ollama tags response from {url}: {error}"
            ))
        })?;
        Ok(payload.models.into_iter().map(|model| model.name).collect())
    }

    pub fn embedding_model_is_available(&self) -> Result<bool, FractalError> {
        Ok(self
            .available_models()?
            .into_iter()
            .any(|name| name == self.config.embedding_model))
    }

    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, FractalError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/api/embed", self.config.base_url.trim_end_matches('/'));
        let request = OllamaEmbedRequest {
            model: self.config.embedding_model.clone(),
            input: texts.to_vec(),
        };
        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to request Ollama embeddings from {url}: {error}"
                ))
            })?
            .error_for_status()
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "Ollama embedding request failed for {url}: {error}"
                ))
            })?;
        let payload: OllamaEmbedResponse = response.json().map_err(|error| {
            FractalError::InvalidState(format!(
                "failed to decode Ollama embedding response from {url}: {error}"
            ))
        })?;
        if payload.embeddings.len() != texts.len() {
            return Err(FractalError::InvalidState(format!(
                "Ollama embedding response count mismatch: expected {} embeddings, got {}",
                texts.len(),
                payload.embeddings.len()
            )));
        }
        Ok(payload.embeddings)
    }

    pub fn benchmark_overlay_roundtrip(
        &self,
        request: OverlayBenchmarkRequest<'_>,
    ) -> Result<OverlayClientBenchmarkResult, FractalError> {
        self.benchmark_overlay_roundtrip_with_config(
            request,
            &OllamaClientBenchmarkConfig::default(),
        )
    }

    pub fn benchmark_overlay_roundtrip_with_config(
        &self,
        request: OverlayBenchmarkRequest<'_>,
        benchmark_config: &OllamaClientBenchmarkConfig,
    ) -> Result<OverlayClientBenchmarkResult, FractalError> {
        validate_benchmark_config(benchmark_config)?;
        let prepared = prepare_overlay_benchmark(request)?;

        warm_embedding_requests(
            self,
            &prepared.canonical_texts,
            &prepared.materialized_texts,
            benchmark_config.warmup_rounds,
        )?;
        let (base, overlay) = measure_interleaved_embedding_requests(
            self,
            &prepared.canonical_texts,
            &prepared.materialized_texts,
            benchmark_config.measure_rounds,
        )?;
        validate_embedding_dimensions(&base, &overlay)?;

        Ok(prepared.summary.finalize(base, overlay.request_ms))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OllamaEmbeddingRequestMetrics {
    pub document_count: usize,
    pub input_bytes: usize,
    pub request_ms: f64,
    pub request_min_ms: f64,
    pub request_max_ms: f64,
    pub embedding_dim: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayClientBenchmarkResult {
    pub document_count: usize,
    pub input_bytes: usize,
    pub canonical_token_count: usize,
    pub transport_symbols: f64,
    pub transport_ratio: f64,
    pub definition_overhead_rate: f64,
    pub embedding_dim: usize,
    pub base: OllamaEmbeddingRequestMetrics,
    pub base_prepare_ms: f64,
    pub overlay_discovery_ms: f64,
    pub overlay_pack_ms: f64,
    pub overlay_materialize_ms: f64,
    pub overlay_request_ms: f64,
}

impl OverlayClientBenchmarkResult {
    pub fn overlay_client_overhead_ms(&self) -> f64 {
        self.overlay_discovery_ms + self.overlay_pack_ms + self.overlay_materialize_ms
    }

    pub fn overlay_extra_client_overhead_ms(&self) -> f64 {
        self.overlay_client_overhead_ms() - self.base_prepare_ms
    }

    pub fn base_total_ms(&self) -> f64 {
        self.base_prepare_ms + self.base.request_ms
    }

    pub fn overlay_total_ms(&self) -> f64 {
        self.overlay_client_overhead_ms() + self.overlay_request_ms
    }

    pub fn request_delta_ms(&self) -> f64 {
        self.overlay_request_ms - self.base.request_ms
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedOverlayBenchmarkSummary {
    document_count: usize,
    input_bytes: usize,
    canonical_token_count: usize,
    transport_symbols: f64,
    transport_ratio: f64,
    definition_overhead_rate: f64,
    base_prepare_ms: f64,
    overlay_discovery_ms: f64,
    overlay_pack_ms: f64,
    overlay_materialize_ms: f64,
}

impl PreparedOverlayBenchmarkSummary {
    fn finalize(
        self,
        base: OllamaEmbeddingRequestMetrics,
        overlay_request_ms: f64,
    ) -> OverlayClientBenchmarkResult {
        OverlayClientBenchmarkResult {
            document_count: self.document_count,
            input_bytes: self.input_bytes,
            canonical_token_count: self.canonical_token_count,
            transport_symbols: self.transport_symbols,
            transport_ratio: self.transport_ratio,
            definition_overhead_rate: self.definition_overhead_rate,
            embedding_dim: base.embedding_dim,
            base,
            base_prepare_ms: self.base_prepare_ms,
            overlay_discovery_ms: self.overlay_discovery_ms,
            overlay_pack_ms: self.overlay_pack_ms,
            overlay_materialize_ms: self.overlay_materialize_ms,
            overlay_request_ms,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct PreparedOverlayBenchmark {
    canonical_texts: Vec<String>,
    materialized_texts: Vec<String>,
    summary: PreparedOverlayBenchmarkSummary,
}

struct PreparedOverlayTransport {
    overlay_batch: OverlayModelFacingBatch,
    transport: crate::OverlayTransportBatch,
    overlay_discovery_ms: f64,
    overlay_pack_ms: f64,
}

fn prepare_overlay_benchmark(
    request: OverlayBenchmarkRequest<'_>,
) -> Result<PreparedOverlayBenchmark, FractalError> {
    let base_prepare_started = Instant::now();
    let canonical_texts = canonicalize_texts(request.tokenizer, request.texts)?;
    let base_prepare_ms = elapsed_ms(base_prepare_started);

    let prepared_transport = prepare_overlay_transport(&request)?;
    let materialize_started = Instant::now();
    let materialized_texts =
        materialize_overlay_texts(request.tokenizer, &prepared_transport.transport)?;
    let overlay_materialize_ms = elapsed_ms(materialize_started);
    if materialized_texts != canonical_texts {
        return Err(FractalError::InvalidState(
            "overlay materialized texts differ from canonical tokenizer materialization"
                .to_string(),
        ));
    }

    Ok(PreparedOverlayBenchmark {
        canonical_texts,
        materialized_texts,
        summary: build_prepared_overlay_summary(
            request.texts,
            &prepared_transport,
            base_prepare_ms,
            overlay_materialize_ms,
        ),
    })
}

fn validate_embedding_dimensions(
    base: &OllamaEmbeddingRequestMetrics,
    overlay: &OllamaEmbeddingRequestMetrics,
) -> Result<(), FractalError> {
    if overlay.embedding_dim == base.embedding_dim {
        return Ok(());
    }
    Err(FractalError::InvalidState(format!(
        "Ollama embedding dimension mismatch between base ({}) and overlay ({}) paths",
        base.embedding_dim, overlay.embedding_dim
    )))
}

fn prepare_overlay_transport(
    request: &OverlayBenchmarkRequest<'_>,
) -> Result<PreparedOverlayTransport, FractalError> {
    let discovery_started = Instant::now();
    let overlay_batch = build_overlay_batch(
        request.tokenizer,
        request.texts,
        request.overlay_mode,
        request.overlay_config,
    )?;
    let overlay_discovery_ms = elapsed_ms(discovery_started);

    let pack_started = Instant::now();
    let transport = OverlayTransportAdapter::new(request.transport_config.clone())
        .prepare_batch(&overlay_batch)?;
    let overlay_pack_ms = elapsed_ms(pack_started);

    Ok(PreparedOverlayTransport {
        overlay_batch,
        transport,
        overlay_discovery_ms,
        overlay_pack_ms,
    })
}

fn build_prepared_overlay_summary(
    texts: &[String],
    prepared_transport: &PreparedOverlayTransport,
    base_prepare_ms: f64,
    overlay_materialize_ms: f64,
) -> PreparedOverlayBenchmarkSummary {
    let transport_summary = prepared_transport.transport.summary();
    PreparedOverlayBenchmarkSummary {
        document_count: texts.len(),
        input_bytes: texts.iter().map(|text| text.len()).sum(),
        canonical_token_count: prepared_transport
            .overlay_batch
            .documents()
            .iter()
            .map(OverlayModelFacingDocument::canonical_token_count)
            .sum(),
        transport_symbols: transport_summary.transport_symbols,
        transport_ratio: transport_summary.transport_ratio(),
        definition_overhead_rate: transport_summary.definition_overhead_rate(),
        base_prepare_ms,
        overlay_discovery_ms: prepared_transport.overlay_discovery_ms,
        overlay_pack_ms: prepared_transport.overlay_pack_ms,
        overlay_materialize_ms,
    }
}

fn build_overlay_batch(
    tokenizer: &HuggingFaceNativeTokenizer,
    texts: &[String],
    overlay_mode: RecursiveOverlayMode,
    overlay_config: &RecursiveOverlayConfig,
) -> Result<OverlayModelFacingBatch, FractalError> {
    texts
        .iter()
        .map(|text| {
            let canonical = tokenizer
                .tokenize_with_byte_offsets(text)
                .map_err(|error| FractalError::InvalidState(error.to_string()))?;
            let overlay = build_recursive_overlay(text, canonical, overlay_mode, overlay_config);
            OverlayModelFacingDocument::new(overlay)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(OverlayModelFacingBatch::new)
}

fn materialize_overlay_texts(
    tokenizer: &HuggingFaceNativeTokenizer,
    transport: &crate::OverlayTransportBatch,
) -> Result<Vec<String>, FractalError> {
    transport
        .expanded_token_ids_by_document()?
        .into_iter()
        .map(|token_ids| {
            tokenizer
                .decode_token_ids(&token_ids)
                .map_err(|error| FractalError::InvalidState(error.to_string()))
        })
        .collect()
}

fn canonicalize_texts(
    tokenizer: &HuggingFaceNativeTokenizer,
    texts: &[String],
) -> Result<Vec<String>, FractalError> {
    texts
        .iter()
        .map(|text| {
            let canonical = tokenizer
                .tokenize_with_byte_offsets(text)
                .map_err(|error| FractalError::InvalidState(error.to_string()))?;
            tokenizer
                .decode_token_ids(&canonical.token_ids)
                .map_err(|error| FractalError::InvalidState(error.to_string()))
        })
        .collect()
}

fn measure_interleaved_embedding_requests(
    client: &OllamaEmbeddingClient,
    base_texts: &[String],
    overlay_texts: &[String],
    measure_rounds: usize,
) -> Result<(OllamaEmbeddingRequestMetrics, OllamaEmbeddingRequestMetrics), FractalError> {
    let mut base_samples = Vec::with_capacity(measure_rounds);
    let mut overlay_samples = Vec::with_capacity(measure_rounds);
    let mut base_embedding_dim = 0usize;
    let mut overlay_embedding_dim = 0usize;

    for round in 0..measure_rounds {
        let base_first = round % 2 == 0;
        if base_first {
            let (request_ms, embedding_dim) = measure_single_embedding_request(client, base_texts)?;
            base_samples.push(request_ms);
            base_embedding_dim = embedding_dim;
            let (request_ms, embedding_dim) =
                measure_single_embedding_request(client, overlay_texts)?;
            overlay_samples.push(request_ms);
            overlay_embedding_dim = embedding_dim;
        } else {
            let (request_ms, embedding_dim) =
                measure_single_embedding_request(client, overlay_texts)?;
            overlay_samples.push(request_ms);
            overlay_embedding_dim = embedding_dim;
            let (request_ms, embedding_dim) = measure_single_embedding_request(client, base_texts)?;
            base_samples.push(request_ms);
            base_embedding_dim = embedding_dim;
        }
    }

    Ok((
        build_request_metrics(base_texts, base_samples, base_embedding_dim),
        build_request_metrics(overlay_texts, overlay_samples, overlay_embedding_dim),
    ))
}

fn build_request_metrics(
    texts: &[String],
    mut samples: Vec<f64>,
    embedding_dim: usize,
) -> OllamaEmbeddingRequestMetrics {
    samples.sort_by(f64::total_cmp);
    OllamaEmbeddingRequestMetrics {
        document_count: texts.len(),
        input_bytes: texts.iter().map(|text| text.len()).sum(),
        request_ms: median_ms(&samples),
        request_min_ms: samples.first().copied().unwrap_or(0.0),
        request_max_ms: samples.last().copied().unwrap_or(0.0),
        embedding_dim,
    }
}

fn measure_single_embedding_request(
    client: &OllamaEmbeddingClient,
    texts: &[String],
) -> Result<(f64, usize), FractalError> {
    let started = Instant::now();
    let embeddings = client.embed_texts(texts)?;
    let request_ms = elapsed_ms(started);
    let embedding_dim = embeddings.first().map_or(0, Vec::len);
    Ok((request_ms, embedding_dim))
}

fn warm_embedding_requests(
    client: &OllamaEmbeddingClient,
    base_texts: &[String],
    overlay_texts: &[String],
    warmup_rounds: usize,
) -> Result<(), FractalError> {
    for round in 0..warmup_rounds {
        if round % 2 == 0 {
            client.embed_texts(base_texts)?;
            client.embed_texts(overlay_texts)?;
        } else {
            client.embed_texts(overlay_texts)?;
            client.embed_texts(base_texts)?;
        }
    }
    Ok(())
}

fn validate_benchmark_config(config: &OllamaClientBenchmarkConfig) -> Result<(), FractalError> {
    if config.measure_rounds == 0 {
        return Err(FractalError::InvalidConfig(
            "Ollama benchmark measure_rounds must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn median_ms(samples: &[f64]) -> f64 {
    samples[samples.len() / 2]
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTagModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaTagModel {
    name: String,
}

#[derive(Debug, Serialize)]
struct OllamaEmbedRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}
