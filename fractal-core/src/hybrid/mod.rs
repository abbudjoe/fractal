pub mod model;
pub mod rescue_attention;
pub mod retrieval_gather;

pub use model::{
    FractalHybridRescuePrevalidationModel, HybridModelShape, HybridRescueForwardOutput,
    HybridRescuePrevalidationMode, HybridRescueStepOutput, PreparedHybridRescueStep,
};
pub use rescue_attention::{
    BaselineRescueAttentionBlock, BaselineRescueAttentionConfig, RescueAttentionBlock,
    RescueAttentionDiagnostics, RescueAttentionInput, RescueAttentionOutput, RescueAttentionShape,
    PHASE1_LEAF_SIZE, PHASE1_LOCAL_WINDOW_SIZE, PHASE1_REMOTE_TOKEN_BUDGET,
    PHASE1_ROUTED_SPAN_COUNT, PHASE1_TOTAL_TOKEN_BUDGET,
};
pub use retrieval_gather::{
    GatheredCandidateRecall, GatheredRetrievalContext, GatheredRetrievalContextShape,
    GatheredRetrievalLayout, GatheredRetrievalProvenance, SealedTokenStateStore,
};
