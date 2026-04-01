use burn::tensor::backend::Backend;
use fractal_core::error::FractalError;

use crate::{
    tokenizer::p1_dynamic_lever_factory, PrimitiveFactory, PrimitiveRunSummary, RecursiveTokenizer,
    TokenizerConfig,
};

use super::{
    fallback::encode_summary_document, EncodedDocument, FaceoffEmissionPolicy, FaceoffFallbackMode,
    FaceoffLocalCacheMode, FaceoffVocab, FaceoffVocabConfig,
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
        self.summarize_with_factory(text, device, p1_dynamic_lever_factory::<B>())
    }

    pub fn summarize_with_factory<B: Backend>(
        &self,
        text: &str,
        device: &B::Device,
        factory: PrimitiveFactory<B>,
    ) -> Result<PrimitiveRunSummary, FractalError> {
        self.tokenizer.run_factory(text, device, factory)
    }

    pub fn induce_vocab_from_texts<B: Backend>(
        &self,
        texts: &[&str],
        device: &B::Device,
    ) -> Result<FaceoffVocab, FractalError> {
        self.induce_vocab_from_texts_with_config::<B>(texts, device, FaceoffVocabConfig::default())
    }

    pub fn induce_vocab_from_texts_for_factory<B: Backend>(
        &self,
        texts: &[&str],
        device: &B::Device,
        factory: PrimitiveFactory<B>,
    ) -> Result<FaceoffVocab, FractalError> {
        self.induce_vocab_from_texts_for_factory_with_config::<B>(
            texts,
            device,
            factory,
            FaceoffVocabConfig::default(),
        )
    }

    pub fn induce_vocab_from_texts_with_config<B: Backend>(
        &self,
        texts: &[&str],
        device: &B::Device,
        config: FaceoffVocabConfig,
    ) -> Result<FaceoffVocab, FractalError> {
        let summaries = texts
            .iter()
            .map(|text| self.summarize_v2::<B>(text, device))
            .collect::<Result<Vec<_>, _>>()?;
        FaceoffVocab::from_summaries_with_config(summaries.iter(), config)
    }

    pub fn induce_vocab_from_texts_for_factory_with_config<B: Backend>(
        &self,
        texts: &[&str],
        device: &B::Device,
        factory: PrimitiveFactory<B>,
        config: FaceoffVocabConfig,
    ) -> Result<FaceoffVocab, FractalError> {
        let summaries = texts
            .iter()
            .map(|text| self.summarize_with_factory::<B>(text, device, factory.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        FaceoffVocab::from_summaries_with_config(summaries.iter(), config)
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
        self.encode_text_v2_with_policy_and_fallback_mode::<B>(
            text,
            vocab,
            device,
            policy,
            FaceoffFallbackMode::Full,
        )
    }

    pub fn encode_text_v2_with_policy_and_fallback_mode<B: Backend>(
        &self,
        text: &str,
        vocab: &FaceoffVocab,
        device: &B::Device,
        policy: FaceoffEmissionPolicy,
        fallback_mode: FaceoffFallbackMode,
    ) -> Result<EncodedDocument, FractalError> {
        self.encode_text_v2_with_policy_and_fallback_mode_and_local_cache_mode::<B>(
            text,
            vocab,
            device,
            policy,
            fallback_mode,
            FaceoffLocalCacheMode::Off,
        )
    }

    pub fn encode_text_v2_with_policy_and_fallback_mode_and_local_cache_mode<B: Backend>(
        &self,
        text: &str,
        vocab: &FaceoffVocab,
        device: &B::Device,
        policy: FaceoffEmissionPolicy,
        fallback_mode: FaceoffFallbackMode,
        local_cache_mode: FaceoffLocalCacheMode,
    ) -> Result<EncodedDocument, FractalError> {
        self.encode_text_with_factory_and_policy(
            text,
            vocab,
            device,
            p1_dynamic_lever_factory::<B>(),
            policy,
            fallback_mode,
            local_cache_mode,
        )
    }

    pub fn encode_text_with_factory_and_policy<B: Backend>(
        &self,
        text: &str,
        vocab: &FaceoffVocab,
        device: &B::Device,
        factory: PrimitiveFactory<B>,
        policy: FaceoffEmissionPolicy,
        fallback_mode: FaceoffFallbackMode,
        local_cache_mode: FaceoffLocalCacheMode,
    ) -> Result<EncodedDocument, FractalError> {
        let summary = self.summarize_with_factory::<B>(text, device, factory)?;
        self.encode_summary_with_policy_and_fallback_mode(
            text,
            &summary,
            vocab,
            policy,
            fallback_mode,
            local_cache_mode,
        )
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
        self.encode_summary_with_policy_and_fallback_mode(
            text,
            summary,
            vocab,
            policy,
            FaceoffFallbackMode::Full,
            FaceoffLocalCacheMode::Off,
        )
    }

    pub fn encode_summary_with_policy_and_fallback_mode(
        &self,
        text: &str,
        summary: &PrimitiveRunSummary,
        vocab: &FaceoffVocab,
        policy: FaceoffEmissionPolicy,
        fallback_mode: FaceoffFallbackMode,
        local_cache_mode: FaceoffLocalCacheMode,
    ) -> Result<EncodedDocument, FractalError> {
        encode_summary_document(
            text,
            summary,
            vocab,
            policy,
            fallback_mode,
            local_cache_mode,
        )
    }

    pub fn decode_document(&self, encoded: &EncodedDocument) -> Result<String, FractalError> {
        encoded.decode()
    }
}
