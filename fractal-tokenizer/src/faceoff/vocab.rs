use std::{
    collections::{BTreeMap, BTreeSet},
    convert::TryFrom,
    fs,
    path::Path,
};

use fractal_core::error::FractalError;
use serde::{Deserialize, Serialize};

use crate::{PrimitiveRunSummary, TokenRecord};

use super::FaceoffTokenId;

pub const FACEOFF_VOCAB_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabEntry {
    pub id: FaceoffTokenId,
    pub digest: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceoffVocab {
    motif_to_id: BTreeMap<String, FaceoffTokenId>,
    id_to_motif: Vec<String>,
    byte_fallback_base: u32,
}

impl FaceoffVocab {
    pub const FORMAT_VERSION: u32 = FACEOFF_VOCAB_FORMAT_VERSION;

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

    pub fn byte_fallback_base(&self) -> u32 {
        self.byte_fallback_base
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), FractalError> {
        let persisted = PersistedFaceoffVocab {
            version: Self::FORMAT_VERSION,
            byte_fallback_base: self.byte_fallback_base,
            motifs: self.id_to_motif.clone(),
        };
        let json = serde_json::to_string_pretty(&persisted).map_err(|source| {
            FractalError::InvalidState(format!("failed to serialize faceoff vocab: {source}"))
        })?;
        fs::write(path.as_ref(), json).map_err(|source| {
            FractalError::InvalidState(format!(
                "failed to write faceoff vocab to `{}`: {source}",
                path.as_ref().display()
            ))
        })?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, FractalError> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|source| {
            FractalError::InvalidState(format!(
                "failed to read faceoff vocab from `{}`: {source}",
                path.display()
            ))
        })?;
        let persisted: PersistedFaceoffVocab = serde_json::from_str(&content).map_err(|source| {
            FractalError::InvalidState(format!(
                "failed to parse faceoff vocab from `{}`: {source}",
                path.display()
            ))
        })?;
        validate_persisted_vocab(&persisted)?;
        Ok(Self::from_sorted_digests(persisted.motifs))
    }

    fn from_sorted_digests(sorted_digests: Vec<String>) -> Self {
        let motif_to_id = sorted_digests
            .iter()
            .enumerate()
            .map(|(index, digest)| (digest.clone(), FaceoffTokenId::new(index as u32)))
            .collect::<BTreeMap<_, _>>();
        let byte_fallback_base = u32::try_from(sorted_digests.len())
            .expect("faceoff vocab motif count should fit within u32");
        Self {
            motif_to_id,
            id_to_motif: sorted_digests,
            byte_fallback_base,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct PersistedFaceoffVocab {
    version: u32,
    byte_fallback_base: u32,
    motifs: Vec<String>,
}

fn validate_persisted_vocab(persisted: &PersistedFaceoffVocab) -> Result<(), FractalError> {
    if persisted.version != FACEOFF_VOCAB_FORMAT_VERSION {
        return Err(FractalError::InvalidState(format!(
            "unsupported faceoff vocab version {} (expected {})",
            persisted.version, FACEOFF_VOCAB_FORMAT_VERSION
        )));
    }
    if persisted.motifs.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(FractalError::InvalidState(
            "persisted faceoff vocab motifs must be sorted and unique".to_string(),
        ));
    }
    if persisted.byte_fallback_base != persisted.motifs.len() as u32 {
        return Err(FractalError::InvalidState(format!(
            "persisted byte fallback base {} does not match motif count {}",
            persisted.byte_fallback_base,
            persisted.motifs.len()
        )));
    }
    Ok(())
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
