use burn::{
    module::Module,
    nn::{loss::CrossEntropyLoss, Embedding, EmbeddingConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Bool, ElementConversion, Int, Tensor},
};

use crate::{
    data_generator::TokenBatch, error::FractalError, router::EarlyExitRouter,
    rule_trait::FractalRule, state::FractalState,
};

#[derive(Module, Debug)]
pub struct FractalModel<B: Backend, R: Module<B>> {
    pub embedding: Embedding<B>,
    pub rule: R,
    pub router: EarlyExitRouter<B>,
    pub output: Linear<B>,
    hidden_dim: usize,
    vocab_size: usize,
    max_recursion_depth: usize,
    pad_token: usize,
}

impl<B: Backend, R: FractalRule<B> + Module<B>> FractalModel<B, R> {
    pub fn new(
        vocab_size: usize,
        hidden_dim: usize,
        max_recursion_depth: usize,
        router_threshold: f32,
        pad_token: usize,
        rule: R,
        device: &B::Device,
    ) -> Self {
        let readout_width = rule.state_layout().readout_width(hidden_dim);

        Self {
            embedding: EmbeddingConfig::new(vocab_size, hidden_dim).init(device),
            router: EarlyExitRouter::new(readout_width, router_threshold, device),
            output: LinearConfig::new(readout_width, vocab_size).init(device),
            rule,
            hidden_dim,
            vocab_size,
            max_recursion_depth,
            pad_token,
        }
    }

    pub fn forward_tokens(
        &self,
        input_ids: Tensor<B, 2, Int>,
    ) -> Result<Tensor<B, 3>, FractalError> {
        self.forward_tokens_with_controls(input_ids, None, true)
    }

    pub fn forward_tokens_with_controls(
        &self,
        input_ids: Tensor<B, 2, Int>,
        force_depth: Option<usize>,
        use_router: bool,
    ) -> Result<Tensor<B, 3>, FractalError> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();
        let embeddings = self.embedding.forward(input_ids);
        let mut state = FractalState::zeros(
            self.rule.state_layout(),
            batch_size,
            self.hidden_dim,
            &device,
        )?;
        let mut logits = Vec::with_capacity(seq_len);

        for position in 0..seq_len {
            let x_t = embeddings
                .clone()
                .narrow(1, position, 1)
                .reshape([batch_size, self.hidden_dim]);
            state = self.recurse_token(state, x_t, force_depth, use_router)?;
            let token_logits =
                self.output
                    .forward(state.readout())
                    .reshape([batch_size, 1, self.vocab_size]);
            logits.push(token_logits);
        }

        Ok(Tensor::cat(logits, 1))
    }

    pub fn loss(
        &self,
        batch: &TokenBatch<B>,
        criterion: &CrossEntropyLoss<B>,
        force_depth: Option<usize>,
        use_router: bool,
    ) -> Result<Tensor<B, 1>, FractalError> {
        let logits =
            self.forward_tokens_with_controls(batch.input_ids.clone(), force_depth, use_router)?;
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let targets = batch.target_ids.clone().reshape([batch_size * seq_len]);
        Ok(criterion.forward(logits.reshape([batch_size * seq_len, vocab_size]), targets))
    }

    pub fn pad_token(&self) -> usize {
        self.pad_token
    }

    fn recurse_token(
        &self,
        mut state: FractalState<B>,
        x: Tensor<B, 2>,
        force_depth: Option<usize>,
        use_router: bool,
    ) -> Result<FractalState<B>, FractalError> {
        let depth_limit = force_depth.unwrap_or(self.max_recursion_depth);
        if !use_router {
            for _ in 0..depth_limit {
                state = self.rule.apply(&state, &x)?;
            }
            return Ok(state);
        }

        let batch_size = state.batch_size();
        let device = x.device();
        let mut active_mask = Tensor::<B, 1, Bool>::ones([batch_size], &device);

        for _ in 0..depth_limit {
            let next_state = self.rule.apply(&state, &x)?;
            state = state.batch_mask_where(active_mask.clone(), next_state)?;
            let exit_mask = self.router.exit_mask(state.readout());
            active_mask = active_mask.clone().bool_and(exit_mask.bool_not());
            if !mask_has_true(active_mask.clone())? {
                break;
            }
        }

        Ok(state)
    }
}

fn mask_has_true<B: Backend>(mask: Tensor<B, 1, Bool>) -> Result<bool, FractalError> {
    Ok(mask.any().int().sum().into_scalar().elem::<i64>() > 0)
}
