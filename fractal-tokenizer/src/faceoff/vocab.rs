use std::collections::{BTreeMap, BTreeSet};

use fractal_core::error::FractalError;

use crate::{PrimitiveRunSummary, TokenRecord};

use super::FaceoffTokenId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabEntry {
    pub id: FaceoffTokenId,
    pub digest: String,
}

#[derive(Clone, Debug)]
pub struct FaceoffVocab {
    motif_to_id: BTreeMap<String, FaceoffTokenId>,
    id_to_motif: Vec<String>,
    byte_fallback_base: u32,
}

impl FaceoffVocab {
    pub fn from_summaries<'a, I>(summaries: I) -> Result<Self, FractalError>
    where
        I: IntoIterator<Item = &'a PrimitiveRunSummary>,
    {
        let digests = summaries
            .into_iter()
            .flat_map(|summary| summary.tokens.iter())
            .map(|record| token_digest(&record.token).map(str::to_owned))
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Self::from_sorted_digests(digests.into_iter().collect()))
    }

    pub fn from_token_records<'a, I>(records: I) -> Result<Self, FractalError>
    where
        I: IntoIterator<Item = &'a TokenRecord>,
    {
        let digests = records
            .into_iter()
            .map(|record| token_digest(&record.token).map(str::to_owned))
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Self::from_sorted_digests(digests.into_iter().collect()))
    }

    pub fn motif_count(&self) -> usize {
        self.id_to_motif.len()
    }

    pub fn entries(&self) -> Vec<VocabEntry> {
        self.id_to_motif
            .iter()
            .enumerate()
            .map(|(id, digest)| VocabEntry {
                id: FaceoffTokenId::new(id as u32),
                digest: digest.clone(),
            })
            .collect()
    }

    pub fn motif_id(&self, digest: &str) -> Option<FaceoffTokenId> {
        self.motif_to_id.get(digest).copied()
    }

    pub fn motif_digest(&self, id: FaceoffTokenId) -> Option<&str> {
        let index = id.as_u32() as usize;
        self.id_to_motif.get(index).map(String::as_str)
    }

    pub fn byte_id(&self, value: u8) -> FaceoffTokenId {
        FaceoffTokenId::new(self.byte_fallback_base + u32::from(value))
    }

    pub fn decode_byte_id(&self, id: FaceoffTokenId) -> Option<u8> {
        let raw = id.as_u32();
        if raw < self.byte_fallback_base {
            return None;
        }
        let offset = raw - self.byte_fallback_base;
        if offset > u8::MAX as u32 {
            return None;
        }
        Some(offset as u8)
    }

    fn from_sorted_digests(sorted_digests: Vec<String>) -> Self {
        let motif_to_id = sorted_digests
            .iter()
            .enumerate()
            .map(|(index, digest)| (digest.clone(), FaceoffTokenId::new(index as u32)))
            .collect::<BTreeMap<_, _>>();
        let byte_fallback_base = sorted_digests.len() as u32;
        Self {
            motif_to_id,
            id_to_motif: sorted_digests,
            byte_fallback_base,
        }
    }
}

pub(crate) fn token_digest(token: &str) -> Result<&str, FractalError> {
    token
        .rsplit_once('-')
        .map(|(_, digest)| digest)
        .ok_or_else(|| {
            FractalError::InvalidState(format!(
                "token `{token}` is missing digest suffix in expected dX-nY-digest format"
            ))
        })
}
