use std::time::Duration;

use fractal_core::error::FractalError;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

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
