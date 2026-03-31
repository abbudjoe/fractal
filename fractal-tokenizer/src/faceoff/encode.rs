use burn::tensor::backend::Backend;
use fractal_core::error::FractalError;

use crate::{
    tokenizer::p1_dynamic_lever_factory, PrimitiveRunSummary, RecursiveTokenizer, TokenizerConfig,
};

use super::{
    fallback::encode_summary_document, EncodedDocument, FaceoffEmissionPolicy, FaceoffVocab,
};

pub struct FaceoffTokenizer {
    tokenizer: RecursiveTokenizer,
    config: TokenizerConfig,
}

impl FaceoffTokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            tokenizer: RecursiveTokenizer::new(config),
            config,
        }
    }

    pub fn config(&self) -> TokenizerConfig {
        self.config
    }

    pub fn summarize_v2<B: Backend>(
        &self,
        text: &str,
        device: &B::Device,
    ) -> Result<PrimitiveRunSummary, FractalError> {
        self.tokenizer
            .run_factory(text, device, p1_dynamic_lever_factory::<B>())
    }

    pub fn induce_vocab_from_texts<B: Backend>(
        &self,
        texts: &[&str],
        device: &B::Device,
    ) -> Result<FaceoffVocab, FractalError> {
        let summaries = texts
            .iter()
            .map(|text| self.summarize_v2::<B>(text, device))
            .collect::<Result<Vec<_>, _>>()?;
        FaceoffVocab::from_summaries(summaries.iter())
    }

    pub fn encode_text_v2<B: Backend>(
        &self,
        text: &str,
        vocab: &FaceoffVocab,
        device: &B::Device,
    ) -> Result<EncodedDocument, FractalError> {
        self.encode_text_v2_with_policy::<B>(
            text,
            vocab,
            device,
            FaceoffEmissionPolicy::FinestKnown,
        )
    }

    pub fn encode_text_v2_with_policy<B: Backend>(
        &self,
        text: &str,
        vocab: &FaceoffVocab,
        device: &B::Device,
        policy: FaceoffEmissionPolicy,
    ) -> Result<EncodedDocument, FractalError> {
        let summary = self.summarize_v2::<B>(text, device)?;
        self.encode_summary_with_policy(text, &summary, vocab, policy)
    }

    pub fn encode_summary(
        &self,
        text: &str,
        summary: &PrimitiveRunSummary,
        vocab: &FaceoffVocab,
    ) -> Result<EncodedDocument, FractalError> {
        self.encode_summary_with_policy(text, summary, vocab, FaceoffEmissionPolicy::FinestKnown)
    }

    pub fn encode_summary_with_policy(
        &self,
        text: &str,
        summary: &PrimitiveRunSummary,
        vocab: &FaceoffVocab,
        policy: FaceoffEmissionPolicy,
    ) -> Result<EncodedDocument, FractalError> {
        encode_summary_document(text, summary, vocab, policy)
    }

    pub fn decode_document(&self, encoded: &EncodedDocument) -> Result<String, FractalError> {
        encoded.decode()
    }
}
