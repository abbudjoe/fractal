// Naming Convention for tokenizer primitives:
// [base]_[lever-description]_v[version]
// Examples: p1_fractal_hybrid_v1, b1_fractal_gated_dyn-residual-norm_v1
mod primitives;
mod tokenizer;

pub use primitives::{
    B1FractalGated, B3FractalHierarchical, B4Universal, P1FractalHybrid, P2Mandelbrot,
};
pub use tokenizer::{
    revived_primitive_factories, PrimitiveFactory, PrimitiveRunSummary, RecursiveTokenizer,
    TokenRecord, TokenizerConfig,
};

#[cfg(test)]
mod tests;
