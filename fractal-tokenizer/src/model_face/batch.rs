use super::ModelFacingDocument;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelFacingBatch {
    documents: Vec<ModelFacingDocument>,
}

impl ModelFacingBatch {
    pub fn new(documents: Vec<ModelFacingDocument>) -> Self {
        Self { documents }
    }

    pub fn singleton(document: ModelFacingDocument) -> Self {
        Self {
            documents: vec![document],
        }
    }

    pub fn documents(&self) -> &[ModelFacingDocument] {
        &self.documents
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ModelFacingDocument> {
        self.documents.iter()
    }

    pub fn into_documents(self) -> Vec<ModelFacingDocument> {
        self.documents
    }
}

impl From<ModelFacingDocument> for ModelFacingBatch {
    fn from(document: ModelFacingDocument) -> Self {
        Self::singleton(document)
    }
}

impl From<Vec<ModelFacingDocument>> for ModelFacingBatch {
    fn from(documents: Vec<ModelFacingDocument>) -> Self {
        Self::new(documents)
    }
}
