use std::{collections::BTreeSet, time::Instant};

use fractal_core::error::FractalError;
use serde::{Deserialize, Serialize};

use super::{
    HuggingFaceNativeTokenizer, OllamaEmbeddingClient, OllamaGenerationClient,
    OllamaGenerationOptions, OllamaGenerationResult, OverlayTransportBatch,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayServerRequestDocument {
    pub document_id: String,
}

impl OverlayServerRequestDocument {
    pub fn new(document_id: impl Into<String>) -> Result<Self, FractalError> {
        let document_id = document_id.into();
        if document_id.trim().is_empty() {
            return Err(FractalError::InvalidConfig(
                "overlay server request document_id must not be empty".to_string(),
            ));
        }
        Ok(Self { document_id })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OverlayServerPromptFrame {
    pub prefix: String,
    pub suffix: String,
}

impl OverlayServerPromptFrame {
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }
    }

    fn apply(&self, prompt_text: String) -> String {
        format!("{}{}{}", self.prefix, prompt_text, self.suffix)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OverlayServerRequest {
    pub documents: Vec<OverlayServerRequestDocument>,
    pub prompt_frame: Option<OverlayServerPromptFrame>,
    pub transport: OverlayTransportBatch,
}

impl OverlayServerRequest {
    pub fn new(
        documents: Vec<OverlayServerRequestDocument>,
        prompt_frame: Option<OverlayServerPromptFrame>,
        transport: OverlayTransportBatch,
    ) -> Result<Self, FractalError> {
        validate_server_request(&documents, &transport)?;
        Ok(Self {
            documents,
            prompt_frame,
            transport,
        })
    }

    pub fn document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn payload_bytes(&self) -> Result<usize, FractalError> {
        bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map(|payload| payload.len())
            .map_err(|error| {
                FractalError::InvalidState(format!(
                    "failed to serialize overlay server request payload: {error}"
                ))
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OverlayServerPreparedDocument {
    document_id: String,
    prompt_text: String,
}

impl OverlayServerPreparedDocument {
    pub fn document_id(&self) -> &str {
        &self.document_id
    }

    pub fn prompt_text(&self) -> &str {
        &self.prompt_text
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayServerPreparedBatch {
    documents: Vec<OverlayServerPreparedDocument>,
    payload_bytes: usize,
    materialize_ms: f64,
}

impl OverlayServerPreparedBatch {
    pub fn documents(&self) -> &[OverlayServerPreparedDocument] {
        &self.documents
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub fn materialize_ms(&self) -> f64 {
        self.materialize_ms
    }

    pub fn prompt_bytes(&self) -> usize {
        self.documents
            .iter()
            .map(|document| document.prompt_text.len())
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayServerEmbeddingResponse {
    prepared: OverlayServerPreparedBatch,
    embeddings: Vec<Vec<f32>>,
    dispatch_ms: f64,
}

impl OverlayServerEmbeddingResponse {
    pub fn prepared(&self) -> &OverlayServerPreparedBatch {
        &self.prepared
    }

    pub fn embeddings(&self) -> &[Vec<f32>] {
        &self.embeddings
    }

    pub fn dispatch_ms(&self) -> f64 {
        self.dispatch_ms
    }

    pub fn total_ms(&self) -> f64 {
        self.prepared.materialize_ms + self.dispatch_ms
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlayServerGenerationResponse {
    prepared: OverlayServerPreparedDocument,
    payload_bytes: usize,
    materialize_ms: f64,
    generation: OllamaGenerationResult,
}

impl OverlayServerGenerationResponse {
    pub fn prepared(&self) -> &OverlayServerPreparedDocument {
        &self.prepared
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_bytes
    }

    pub fn materialize_ms(&self) -> f64 {
        self.materialize_ms
    }

    pub fn generation(&self) -> &OllamaGenerationResult {
        &self.generation
    }

    pub fn total_ms(&self) -> f64 {
        self.materialize_ms + self.generation.request_ms
    }
}

#[derive(Clone)]
pub struct OverlayEnvelopeServer {
    tokenizer: HuggingFaceNativeTokenizer,
}

impl OverlayEnvelopeServer {
    pub fn new(tokenizer: HuggingFaceNativeTokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn prepare_batch(
        &self,
        request: &OverlayServerRequest,
    ) -> Result<OverlayServerPreparedBatch, FractalError> {
        validate_server_request(&request.documents, &request.transport)?;
        let payload_bytes = request.payload_bytes()?;
        let started = Instant::now();
        let expanded = request.transport.expanded_token_ids_by_document()?;
        let documents = request
            .documents
            .iter()
            .zip(expanded)
            .map(|(document, token_ids)| {
                self.tokenizer
                    .decode_token_ids(&token_ids)
                    .map(|prompt_text| {
                        let prompt_text = request
                            .prompt_frame
                            .as_ref()
                            .map_or(prompt_text.clone(), |frame| frame.apply(prompt_text));
                        OverlayServerPreparedDocument {
                            document_id: document.document_id.clone(),
                            prompt_text,
                        }
                    })
                    .map_err(|error| FractalError::InvalidState(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(OverlayServerPreparedBatch {
            documents,
            payload_bytes,
            materialize_ms: elapsed_ms(started),
        })
    }

    pub fn embed(
        &self,
        request: &OverlayServerRequest,
        client: &OllamaEmbeddingClient,
    ) -> Result<OverlayServerEmbeddingResponse, FractalError> {
        let prepared = self.prepare_batch(request)?;
        let texts = prepared
            .documents
            .iter()
            .map(|document| document.prompt_text.clone())
            .collect::<Vec<_>>();
        let started = Instant::now();
        let embeddings = client.embed_texts(&texts)?;
        let dispatch_ms = elapsed_ms(started);
        if embeddings.len() != texts.len() {
            return Err(FractalError::InvalidState(format!(
                "overlay server embedding response count mismatch: expected {} embeddings, got {}",
                texts.len(),
                embeddings.len()
            )));
        }
        Ok(OverlayServerEmbeddingResponse {
            prepared,
            embeddings,
            dispatch_ms,
        })
    }

    pub fn generate(
        &self,
        request: &OverlayServerRequest,
        client: &OllamaGenerationClient,
        options: &OllamaGenerationOptions,
    ) -> Result<OverlayServerGenerationResponse, FractalError> {
        let prepared = self.prepare_batch(request)?;
        let mut documents = prepared.documents.into_iter();
        let document = documents.next().ok_or_else(|| {
            FractalError::InvalidConfig(
                "overlay server generation requests must contain exactly one document".to_string(),
            )
        })?;
        if documents.next().is_some() {
            return Err(FractalError::InvalidConfig(
                "overlay server generation requests must contain exactly one document".to_string(),
            ));
        }
        let generation = client.generate_text(document.prompt_text(), options)?;
        Ok(OverlayServerGenerationResponse {
            prepared: document,
            payload_bytes: prepared.payload_bytes,
            materialize_ms: prepared.materialize_ms,
            generation,
        })
    }
}

fn validate_server_request(
    documents: &[OverlayServerRequestDocument],
    transport: &OverlayTransportBatch,
) -> Result<(), FractalError> {
    if documents.is_empty() {
        return Err(FractalError::InvalidConfig(
            "overlay server request must contain at least one document".to_string(),
        ));
    }
    if !transport.exact_ok() {
        return Err(FractalError::InvalidState(
            "overlay server request transport must expand exactly back to canonical token ids"
                .to_string(),
        ));
    }
    let transport_documents = transport.document_views().len();
    if documents.len() != transport_documents {
        return Err(FractalError::InvalidConfig(format!(
            "overlay server request document count mismatch: request={} transport={transport_documents}",
            documents.len(),
        )));
    }
    let unique_ids = documents
        .iter()
        .map(|document| document.document_id.as_str())
        .collect::<BTreeSet<_>>();
    if unique_ids.len() != documents.len() {
        return Err(FractalError::InvalidConfig(
            "overlay server request document ids must be unique".to_string(),
        ));
    }
    Ok(())
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}
