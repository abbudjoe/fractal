use burn::{
    module::Module,
    nn::{loss::CrossEntropyLoss, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Bool, ElementConversion, Int, Tensor, TensorData},
};

use crate::{
    data_generator::TokenBatch,
    diagnostics::{
        DiagnosticEventKind, DiagnosticsRecorder, ForwardGraphBurden,
        OutputProjectionDiagnosticContext, ProjectionDiagnosticsSink,
        RuleProjectionDiagnosticContext, TrainStepDiagnosticContext,
    },
    error::FractalError,
    language_model_head::{LanguageModelHead, LanguageModelHeadConfig},
    lifecycle::RunPhase,
    router::EarlyExitRouter,
    rule_trait::{ApplyContext, FractalRule},
    state::FractalState,
};

#[derive(Module, Debug)]
pub struct FractalModel<B: Backend, R: Module<B>> {
    pub embedding: Embedding<B>,
    pub rule: R,
    pub router: EarlyExitRouter<B>,
    pub output: LanguageModelHead<B>,
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
            output: LanguageModelHeadConfig::new(readout_width, vocab_size).init(device),
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
        self.forward_tokens_with_controls(input_ids, None, true, None, None)
            .map(|(logits, _)| logits)
    }

    pub fn forward_tokens_with_controls(
        &self,
        input_ids: Tensor<B, 2, Int>,
        force_depth: Option<usize>,
        use_router: bool,
        diagnostics: Option<&mut DiagnosticsRecorder>,
        step_context: Option<TrainStepDiagnosticContext>,
    ) -> Result<(Tensor<B, 3>, ForwardGraphBurden), FractalError> {
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
        let mut diagnostics = diagnostics;
        let mut graph_burden = ForwardGraphBurdenAccumulator::new(
            batch_size,
            seq_len,
            use_router,
            force_depth,
        );

        for position in 0..seq_len {
            let x_t = embeddings
                .clone()
                .narrow(1, position, 1)
                .reshape([batch_size, self.hidden_dim]);
            let input_shape = x_t.dims().into_iter().collect::<Vec<_>>();
            state = self.recurse_token(
                state,
                x_t,
                RecurseTokenFrame {
                    force_depth,
                    use_router,
                    diagnostics: diagnostics
                        .as_deref_mut()
                        .map(|sink| sink as &mut dyn ProjectionDiagnosticsSink),
                    step_context,
                    position,
                    sequence_length: seq_len,
                    graph_burden: &mut graph_burden,
                },
            )?;
            let readout = state.readout();
            if let (Some(context), Some(recorder)) = (step_context, diagnostics.as_deref_mut()) {
                recorder.emit_forward_position(
                    RunPhase::Train,
                    context,
                    position,
                    seq_len,
                    input_shape,
                    readout.dims().into_iter().collect(),
                )?;
            }
            let output_input_shape = readout.dims().into_iter().collect::<Vec<_>>();
            let output_projection = self.output.forward(readout);
            if let (Some(context), Some(recorder)) = (step_context, diagnostics.as_deref_mut()) {
                recorder.emit_output_projection(
                    RunPhase::Train,
                    OutputProjectionDiagnosticContext {
                        step: context.step,
                        tokens_seen: context.tokens_seen,
                        position,
                        sequence_length: seq_len,
                    },
                    self.output.diagnostic_spec(
                        "fractal_model",
                        "output",
                        output_input_shape,
                        output_projection.dims().into_iter().collect(),
                    ),
                )?;
            }
            let token_logits = output_projection.reshape([batch_size, 1, self.vocab_size]);
            logits.push(token_logits);
        }

        Ok((Tensor::cat(logits, 1), graph_burden.finish()))
    }

    pub fn loss(
        &self,
        batch: &TokenBatch<B>,
        criterion: &CrossEntropyLoss<B>,
        force_depth: Option<usize>,
        use_router: bool,
    ) -> Result<Tensor<B, 1>, FractalError> {
        self.loss_with_diagnostics(batch, criterion, force_depth, use_router, None, None)
    }

    pub fn loss_with_diagnostics(
        &self,
        batch: &TokenBatch<B>,
        criterion: &CrossEntropyLoss<B>,
        force_depth: Option<usize>,
        use_router: bool,
        diagnostics: Option<&mut DiagnosticsRecorder>,
        step_context: Option<TrainStepDiagnosticContext>,
    ) -> Result<Tensor<B, 1>, FractalError> {
        let mut diagnostics = diagnostics;
        if let (Some(context), Some(recorder)) = (step_context, diagnostics.as_mut()) {
            recorder.emit_event(
                RunPhase::Train,
                Some(context.step),
                Some(context.tokens_seen),
                DiagnosticEventKind::ForwardStart {
                    input_shape: batch.input_ids.dims().into_iter().collect(),
                },
            )?;
        }
        let (logits, graph_burden) = self.forward_tokens_with_controls(
            batch.input_ids.clone(),
            force_depth,
            use_router,
            diagnostics.as_deref_mut(),
            step_context,
        )?;
        if let (Some(context), Some(recorder)) = (step_context, diagnostics.as_mut()) {
            recorder.emit_event(
                RunPhase::Train,
                Some(context.step),
                Some(context.tokens_seen),
                DiagnosticEventKind::ForwardComplete {
                    logits_shape: logits.dims().into_iter().collect(),
                    graph_burden: Box::new(graph_burden),
                },
            )?;
            recorder.emit_event(
                RunPhase::Train,
                Some(context.step),
                Some(context.tokens_seen),
                DiagnosticEventKind::LossStart {
                    target_shape: batch.target_ids.dims().into_iter().collect(),
                },
            )?;
        }
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let targets = batch.target_ids.clone().reshape([batch_size * seq_len]);
        let loss = criterion.forward(logits.reshape([batch_size * seq_len, vocab_size]), targets);
        if let (Some(context), Some(recorder)) = (step_context, diagnostics.as_mut()) {
            recorder.emit_event(
                RunPhase::Train,
                Some(context.step),
                Some(context.tokens_seen),
                DiagnosticEventKind::LossComplete {
                    loss_shape: loss.dims().into_iter().collect(),
                },
            )?;
        }
        Ok(loss)
    }

    pub fn pad_token(&self) -> usize {
        self.pad_token
    }

    fn recurse_token(
        &self,
        mut state: FractalState<B>,
        x: Tensor<B, 2>,
        frame: RecurseTokenFrame<'_>,
    ) -> Result<FractalState<B>, FractalError> {
        let RecurseTokenFrame {
            force_depth,
            use_router,
            diagnostics,
            step_context,
            position,
            sequence_length,
            graph_burden,
        } = frame;
        let depth_limit = force_depth.unwrap_or(self.max_recursion_depth);
        let mut diagnostics = diagnostics;
        if !use_router {
            for depth_index in 0..depth_limit {
                graph_burden.record_rule_invocation(depth_index + 1);
                state = self.rule.apply_with_diagnostics(
                    &state,
                    &x,
                    ApplyContext {
                        depth: depth_index + 1,
                        max_depth: depth_limit,
                    },
                    diagnostics
                        .as_deref_mut()
                        .map(|sink| sink as &mut dyn ProjectionDiagnosticsSink),
                    step_context.map(|context| RuleProjectionDiagnosticContext {
                        step: context.step,
                        tokens_seen: context.tokens_seen,
                        position,
                        sequence_length,
                        recursion_depth: depth_index + 1,
                        max_recursion_depth: depth_limit,
                    }),
                )?;
            }
            graph_burden.record_depth_limit_position();
            return Ok(state);
        }

        let batch_size = state.batch_size();
        let device = x.device();
        let mut active_mask = Tensor::<B, 1, Bool>::from_bool(
            TensorData::new(vec![true; batch_size], [batch_size]),
            &device,
        );

        for depth_index in 0..depth_limit {
            graph_burden.record_rule_invocation(depth_index + 1);
            let next_state = self.rule.apply_with_diagnostics(
                &state,
                &x,
                ApplyContext {
                    depth: depth_index + 1,
                    max_depth: depth_limit,
                },
                diagnostics
                    .as_deref_mut()
                    .map(|sink| sink as &mut dyn ProjectionDiagnosticsSink),
                step_context.map(|context| RuleProjectionDiagnosticContext {
                    step: context.step,
                    tokens_seen: context.tokens_seen,
                    position,
                    sequence_length,
                    recursion_depth: depth_index + 1,
                    max_recursion_depth: depth_limit,
                    }),
            )?;
            state = state.batch_mask_where(active_mask.clone(), next_state)?;
            let exit_mask = self.router.exit_mask(state.readout());
            let continuing_mask =
                (active_mask.clone().int() * exit_mask.clone().bool_not().int()).greater_elem(0);
            let exiting_mask = (active_mask.clone().int() * exit_mask.int()).greater_elem(0);
            graph_burden.record_router_progress(
                count_true(exiting_mask)?,
                count_true(continuing_mask.clone())?,
            );
            active_mask = continuing_mask;
            if !mask_has_true(active_mask.clone())? {
                if depth_index + 1 < depth_limit {
                    graph_burden.record_early_exit_position();
                } else {
                    graph_burden.record_depth_limit_position();
                }
                break;
            }
        }

        if mask_has_true(active_mask.clone())? {
            graph_burden.record_depth_limit_position();
        }

        Ok(state)
    }
}

fn mask_has_true<B: Backend>(mask: Tensor<B, 1, Bool>) -> Result<bool, FractalError> {
    Ok(mask.any().into_scalar().elem::<bool>())
}

fn count_true<B: Backend>(mask: Tensor<B, 1, Bool>) -> Result<usize, FractalError> {
    Ok(mask.int().sum().into_scalar().elem::<i64>() as usize)
}

struct RecurseTokenFrame<'a> {
    force_depth: Option<usize>,
    use_router: bool,
    diagnostics: Option<&'a mut dyn ProjectionDiagnosticsSink>,
    step_context: Option<TrainStepDiagnosticContext>,
    position: usize,
    sequence_length: usize,
    graph_burden: &'a mut ForwardGraphBurdenAccumulator,
}

#[derive(Clone, Debug)]
struct ForwardGraphBurdenAccumulator {
    batch_size: usize,
    sequence_length: usize,
    router_enabled: bool,
    forced_depth: Option<usize>,
    positions_processed: usize,
    rule_invocations: usize,
    max_recursion_depth_observed: usize,
    positions_early_exited: usize,
    positions_reached_depth_limit: usize,
    router_exit_elements: usize,
    router_continue_elements: usize,
}

impl ForwardGraphBurdenAccumulator {
    fn new(
        batch_size: usize,
        sequence_length: usize,
        router_enabled: bool,
        forced_depth: Option<usize>,
    ) -> Self {
        Self {
            batch_size,
            sequence_length,
            router_enabled,
            forced_depth,
            positions_processed: 0,
            rule_invocations: 0,
            max_recursion_depth_observed: 0,
            positions_early_exited: 0,
            positions_reached_depth_limit: 0,
            router_exit_elements: 0,
            router_continue_elements: 0,
        }
    }

    fn record_rule_invocation(&mut self, recursion_depth: usize) {
        self.rule_invocations += 1;
        self.max_recursion_depth_observed = self.max_recursion_depth_observed.max(recursion_depth);
    }

    fn record_router_progress(&mut self, exited_elements: usize, continuing_elements: usize) {
        self.router_exit_elements += exited_elements;
        self.router_continue_elements += continuing_elements;
    }

    fn record_early_exit_position(&mut self) {
        self.positions_processed += 1;
        self.positions_early_exited += 1;
    }

    fn record_depth_limit_position(&mut self) {
        self.positions_processed += 1;
        self.positions_reached_depth_limit += 1;
    }

    fn finish(self) -> ForwardGraphBurden {
        ForwardGraphBurden {
            batch_size: self.batch_size,
            sequence_length: self.sequence_length,
            positions_processed: self.positions_processed,
            rule_invocations: self.rule_invocations,
            max_recursion_depth_observed: self.max_recursion_depth_observed,
            router_enabled: self.router_enabled,
            forced_depth: self.forced_depth,
            positions_early_exited: self.positions_early_exited,
            positions_reached_depth_limit: self.positions_reached_depth_limit,
            router_exit_elements: self.router_exit_elements,
            router_continue_elements: self.router_continue_elements,
        }
    }
}
