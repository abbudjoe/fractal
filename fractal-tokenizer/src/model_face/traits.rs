use fractal_core::error::FractalError;

use super::batch::ModelFacingBatch;

pub trait ModelBatch {
    fn documents(&self) -> &[super::ModelFacingDocument];

    fn len(&self) -> usize {
        self.documents().len()
    }

    fn is_empty(&self) -> bool {
        self.documents().is_empty()
    }
}

impl ModelBatch for ModelFacingBatch {
    fn documents(&self) -> &[super::ModelFacingDocument] {
        self.documents()
    }
}

pub trait ModelAdapter {
    type Input;
    type Output;

    fn prepare(&self, input: &Self::Input) -> Result<Self::Output, FractalError>;
}
