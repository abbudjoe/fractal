use crate::error::FractalError;

use super::rescue_attention::RescueAttentionShape;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridRescuePrevalidationMode {
    LocalOnly,
    RoutedRemote,
    OracleRemote,
    OracleRemoteWithOracleExactTokenSubset,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridModelShape {
    pub vocab_size: usize,
    pub token_state_dim: usize,
    pub rescue_attention: RescueAttentionShape,
}

impl HybridModelShape {
    pub fn validate(self) -> Result<Self, FractalError> {
        ensure_nonzero("hybrid_model.vocab_size", self.vocab_size)?;
        ensure_nonzero("hybrid_model.token_state_dim", self.token_state_dim)?;
        ensure_match(
            "hybrid_model.rescue_attention.token_state_dim",
            self.rescue_attention.token_state_dim,
            self.token_state_dim,
        )?;
        self.rescue_attention.validate()?;
        Ok(self)
    }
}

fn ensure_nonzero(name: &str, value: usize) -> Result<(), FractalError> {
    if value == 0 {
        return Err(FractalError::InvalidConfig(format!(
            "{name} must be greater than zero"
        )));
    }

    Ok(())
}

fn ensure_match(name: &str, actual: usize, expected: usize) -> Result<(), FractalError> {
    if actual != expected {
        return Err(FractalError::InvalidConfig(format!(
            "{name} mismatch: expected {expected}, got {actual}"
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_model_shape_rejects_token_width_mismatch() {
        let error = HybridModelShape {
            vocab_size: 257,
            token_state_dim: 64,
            rescue_attention: RescueAttentionShape {
                token_state_dim: 48,
                attention_dim: 32,
                local_window_size: 256,
                routed_span_count: 8,
                leaf_size: 16,
                remote_token_budget: 128,
                sink_token_count: 0,
                total_token_budget: 384,
            },
        }
        .validate()
        .unwrap_err();

        assert!(
            matches!(error, FractalError::InvalidConfig(message) if message.contains("hybrid_model.rescue_attention.token_state_dim"))
        );
    }
}
