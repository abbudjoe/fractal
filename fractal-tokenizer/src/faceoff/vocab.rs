use std::{
    collections::{BTreeMap, BTreeSet},
    convert::TryFrom,
    fs,
    path::Path,
};

use fractal_core::error::FractalError;
use serde::{Deserialize, Serialize};

use crate::{PrimitiveRunSummary, TokenRecord};

use super::{lexeme::lexical_shape_key, FaceoffLexemeKind, FaceoffTokenId};

pub const FACEOFF_VOCAB_FORMAT_VERSION: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceoffVocabConfig {
    pub min_occurrence_count: usize,
    pub min_doc_count: usize,
    pub max_token_bytes: Option<usize>,
    pub min_shape_occurrence_count: usize,
    pub min_shape_doc_count: usize,
    pub min_shape_distinct_text_count: usize,
}

impl Default for FaceoffVocabConfig {
    fn default() -> Self {
        Self {
            min_occurrence_count: 1,
            min_doc_count: 1,
            max_token_bytes: None,
            min_shape_occurrence_count: 2,
            min_shape_doc_count: 2,
            min_shape_distinct_text_count: 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocabEntry {
    pub id: FaceoffTokenId,
    pub digest: String,
    pub text: String,
    pub digests: Vec<String>,
    pub occurrence_count: usize,
    pub doc_count: usize,
    pub min_depth: usize,
    pub max_depth: usize,
    pub max_byte_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeEntry {
    pub id: FaceoffTokenId,
    pub shape: String,
    pub digest: String,
    pub occurrence_count: usize,
    pub doc_count: usize,
    pub distinct_text_count: usize,
    pub max_byte_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrototypeEntry {
    pub id: FaceoffTokenId,
    pub cluster: String,
    pub digest: String,
    pub occurrence_count: usize,
    pub doc_count: usize,
    pub distinct_text_count: usize,
    pub max_byte_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FaceoffVocab {
    motif_to_id: BTreeMap<String, FaceoffTokenId>,
    prototype_to_id: BTreeMap<String, FaceoffTokenId>,
    literal_to_id: BTreeMap<String, FaceoffTokenId>,
    shape_to_id: BTreeMap<String, FaceoffTokenId>,
    entries: Vec<VocabEntry>,
    prototype_entries: Vec<PrototypeEntry>,
    shape_entries: Vec<ShapeEntry>,
    byte_fallback_base: u32,
}

impl FaceoffVocab {
    pub const FORMAT_VERSION: u32 = FACEOFF_VOCAB_FORMAT_VERSION;

    pub fn from_summaries<'a, I>(summaries: I) -> Result<Self, FractalError>
    where
        I: IntoIterator<Item = &'a PrimitiveRunSummary>,
    {
        Self::from_summaries_with_config(summaries, FaceoffVocabConfig::default())
    }

    pub fn from_summaries_with_config<'a, I>(
        summaries: I,
        config: FaceoffVocabConfig,
    ) -> Result<Self, FractalError>
    where
        I: IntoIterator<Item = &'a PrimitiveRunSummary>,
    {
        let mut aggregates = BTreeMap::<String, MotifAggregate>::new();
        let mut prototype_aggregates = BTreeMap::<String, PrototypeAggregate>::new();
        let mut shape_aggregates = BTreeMap::<String, ShapeAggregate>::new();

        for (doc_index, summary) in summaries.into_iter().enumerate() {
            for record in &summary.tokens {
                let aggregate = aggregates.entry(record.text.clone()).or_default();
                aggregate.observe(record, doc_index)?;
                let cluster = prototype_cluster_key(record)?;
                let aggregate = prototype_aggregates.entry(cluster).or_default();
                aggregate.observe(record, &record.text, doc_index)?;
                let shape = lexical_shape_key(&record.text);
                let aggregate = shape_aggregates.entry(shape).or_default();
                aggregate.observe(record, &record.text, doc_index)?;
            }
        }

        let entries = aggregates
            .into_iter()
            .filter_map(|(text, aggregate)| aggregate.into_entry(text, config))
            .collect::<Vec<_>>();
        let prototype_entries = prototype_aggregates
            .into_iter()
            .filter_map(|(cluster, aggregate)| aggregate.into_entry(cluster, config))
            .collect::<Vec<_>>();
        let shape_entries = shape_aggregates
            .into_iter()
            .filter_map(|(shape, aggregate)| aggregate.into_entry(shape, config))
            .collect::<Vec<_>>();
        Ok(Self::from_entries(entries, prototype_entries, shape_entries))
    }

    pub fn from_token_records<'a, I>(records: I) -> Result<Self, FractalError>
    where
        I: IntoIterator<Item = &'a TokenRecord>,
    {
        let mut aggregates = BTreeMap::<String, MotifAggregate>::new();
        let mut prototype_aggregates = BTreeMap::<String, PrototypeAggregate>::new();
        let mut shape_aggregates = BTreeMap::<String, ShapeAggregate>::new();
        for record in records {
            let aggregate = aggregates.entry(record.text.clone()).or_default();
            aggregate.observe(record, 0)?;
            let cluster = prototype_cluster_key(record)?;
            let aggregate = prototype_aggregates.entry(cluster).or_default();
            aggregate.observe(record, &record.text, 0)?;
            let shape = lexical_shape_key(&record.text);
            let aggregate = shape_aggregates.entry(shape).or_default();
            aggregate.observe(record, &record.text, 0)?;
        }
        let entries = aggregates
            .into_iter()
            .filter_map(|(text, aggregate)| {
                aggregate.into_entry(text, FaceoffVocabConfig::default())
            })
            .collect::<Vec<_>>();
        let prototype_entries = prototype_aggregates
            .into_iter()
            .filter_map(|(cluster, aggregate)| {
                aggregate.into_entry(cluster, FaceoffVocabConfig::default())
            })
            .collect::<Vec<_>>();
        let shape_entries = shape_aggregates
            .into_iter()
            .filter_map(|(shape, aggregate)| {
                aggregate.into_entry(shape, FaceoffVocabConfig::default())
            })
            .collect::<Vec<_>>();
        Ok(Self::from_entries(entries, prototype_entries, shape_entries))
    }

    pub fn motif_count(&self) -> usize {
        self.entries.len() + self.prototype_entries.len() + self.shape_entries.len()
    }

    pub fn entries(&self) -> Vec<VocabEntry> {
        self.entries.clone()
    }

    pub fn shape_entries(&self) -> Vec<ShapeEntry> {
        self.shape_entries.clone()
    }

    pub fn prototype_entries(&self) -> Vec<PrototypeEntry> {
        self.prototype_entries.clone()
    }

    pub fn motif_id(&self, digest: &str) -> Option<FaceoffTokenId> {
        self.motif_to_id.get(digest).copied()
    }

    pub(crate) fn prototype_id(&self, record: &TokenRecord) -> Result<Option<FaceoffTokenId>, FractalError> {
        let cluster = prototype_cluster_key(record)?;
        Ok(self.prototype_to_id.get(&cluster).copied())
    }

    pub fn literal_id(&self, text: &str) -> Option<FaceoffTokenId> {
        self.literal_to_id.get(text).copied()
    }

    pub fn shape_id_for_text(&self, text: &str) -> Option<FaceoffTokenId> {
        let shape = lexical_shape_key(text);
        self.shape_to_id.get(&shape).copied()
    }

    pub fn motif_digest(&self, id: FaceoffTokenId) -> Option<&str> {
        let index = id.as_u32() as usize;
        if let Some(entry) = self.entries.get(index) {
            return Some(entry.digest.as_str());
        }
        let prototype_index = index.checked_sub(self.entries.len())?;
        if let Some(entry) = self.prototype_entries.get(prototype_index) {
            return Some(entry.digest.as_str());
        }
        let shape_index = prototype_index.checked_sub(self.prototype_entries.len())?;
        self.shape_entries
            .get(shape_index)
            .map(|entry| entry.digest.as_str())
    }

    pub fn byte_id(&self, value: u8) -> FaceoffTokenId {
        FaceoffTokenId::new(self.byte_fallback_base + u32::from(value))
    }

    pub fn lexical_id(&self, kind: FaceoffLexemeKind) -> FaceoffTokenId {
        FaceoffTokenId::new(self.byte_fallback_base + 256 + kind.stable_index())
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
            motifs: self.entries.iter().map(PersistedVocabEntry::from).collect(),
            prototypes: self
                .prototype_entries
                .iter()
                .map(PersistedPrototypeEntry::from)
                .collect(),
            shapes: self
                .shape_entries
                .iter()
                .map(PersistedShapeEntry::from)
                .collect(),
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
        let persisted: PersistedFaceoffVocab =
            serde_json::from_str(&content).map_err(|source| {
                FractalError::InvalidState(format!(
                    "failed to parse faceoff vocab from `{}`: {source}",
                    path.display()
                ))
            })?;
        validate_persisted_vocab(&persisted)?;
        Ok(Self::from_entries(
            persisted
                .motifs
                .into_iter()
                .enumerate()
                .map(|(index, motif)| VocabEntry {
                    id: FaceoffTokenId::new(index as u32),
                    digest: motif.digest,
                    text: motif.text,
                    digests: motif.digests,
                    occurrence_count: motif.occurrence_count,
                    doc_count: motif.doc_count,
                    min_depth: motif.min_depth,
                    max_depth: motif.max_depth,
                    max_byte_len: motif.max_byte_len,
                })
                .collect(),
            persisted
                .prototypes
                .into_iter()
                .enumerate()
                .map(|(index, prototype)| PrototypeEntry {
                    id: FaceoffTokenId::new(index as u32),
                    cluster: prototype.cluster,
                    digest: prototype.digest,
                    occurrence_count: prototype.occurrence_count,
                    doc_count: prototype.doc_count,
                    distinct_text_count: prototype.distinct_text_count,
                    max_byte_len: prototype.max_byte_len,
                })
                .collect(),
            persisted
                .shapes
                .into_iter()
                .enumerate()
                .map(|(index, shape)| ShapeEntry {
                    id: FaceoffTokenId::new(index as u32),
                    shape: shape.shape,
                    digest: shape.digest,
                    occurrence_count: shape.occurrence_count,
                    doc_count: shape.doc_count,
                    distinct_text_count: shape.distinct_text_count,
                    max_byte_len: shape.max_byte_len,
                })
                .collect(),
        ))
    }

    fn from_entries(
        mut entries: Vec<VocabEntry>,
        mut prototype_entries: Vec<PrototypeEntry>,
        mut shape_entries: Vec<ShapeEntry>,
    ) -> Self {
        entries.sort_by(|left, right| {
            left.text
                .cmp(&right.text)
                .then_with(|| left.digest.cmp(&right.digest))
        });
        let entries = entries
            .into_iter()
            .enumerate()
            .map(|(index, mut entry)| {
                entry.id = FaceoffTokenId::new(index as u32);
                entry
            })
            .collect::<Vec<_>>();
        prototype_entries.sort_by(|left, right| {
            left.cluster
                .cmp(&right.cluster)
                .then_with(|| left.digest.cmp(&right.digest))
        });
        let prototype_entries = prototype_entries
            .into_iter()
            .enumerate()
            .map(|(index, mut entry)| {
                entry.id = FaceoffTokenId::new((entries.len() + index) as u32);
                entry
            })
            .collect::<Vec<_>>();
        shape_entries.sort_by(|left, right| {
            left.shape
                .cmp(&right.shape)
                .then_with(|| left.digest.cmp(&right.digest))
        });
        let shape_entries = shape_entries
            .into_iter()
            .enumerate()
            .map(|(index, mut entry)| {
                entry.id =
                    FaceoffTokenId::new((entries.len() + prototype_entries.len() + index) as u32);
                entry
            })
            .collect::<Vec<_>>();
        let motif_to_id = entries
            .iter()
            .flat_map(|entry| {
                entry
                    .digests
                    .iter()
                    .cloned()
                    .map(move |digest| (digest, entry.id))
            })
            .chain(
                prototype_entries
                    .iter()
                    .map(|entry| (entry.digest.clone(), entry.id)),
            )
            .chain(
                shape_entries
                    .iter()
                    .map(|entry| (entry.digest.clone(), entry.id)),
            )
            .collect::<BTreeMap<_, _>>();
        let prototype_to_id = prototype_entries
            .iter()
            .map(|entry| (entry.cluster.clone(), entry.id))
            .collect::<BTreeMap<_, _>>();
        let literal_to_id = entries
            .iter()
            .map(|entry| (entry.text.clone(), entry.id))
            .collect::<BTreeMap<_, _>>();
        let shape_to_id = shape_entries
            .iter()
            .map(|entry| (entry.shape.clone(), entry.id))
            .collect::<BTreeMap<_, _>>();
        let byte_fallback_base = u32::try_from(
            entries.len() + prototype_entries.len() + shape_entries.len(),
        )
            .expect("faceoff vocab motif count should fit within u32");
        Self {
            motif_to_id,
            prototype_to_id,
            literal_to_id,
            shape_to_id,
            entries,
            prototype_entries,
            shape_entries,
            byte_fallback_base,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct PersistedFaceoffVocab {
    version: u32,
    byte_fallback_base: u32,
    motifs: Vec<PersistedVocabEntry>,
    #[serde(default)]
    prototypes: Vec<PersistedPrototypeEntry>,
    #[serde(default)]
    shapes: Vec<PersistedShapeEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedVocabEntry {
    digest: String,
    text: String,
    digests: Vec<String>,
    occurrence_count: usize,
    doc_count: usize,
    min_depth: usize,
    max_depth: usize,
    max_byte_len: usize,
}

impl From<&VocabEntry> for PersistedVocabEntry {
    fn from(entry: &VocabEntry) -> Self {
        Self {
            digest: entry.digest.clone(),
            text: entry.text.clone(),
            digests: entry.digests.clone(),
            occurrence_count: entry.occurrence_count,
            doc_count: entry.doc_count,
            min_depth: entry.min_depth,
            max_depth: entry.max_depth,
            max_byte_len: entry.max_byte_len,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedPrototypeEntry {
    cluster: String,
    digest: String,
    occurrence_count: usize,
    doc_count: usize,
    distinct_text_count: usize,
    max_byte_len: usize,
}

impl From<&PrototypeEntry> for PersistedPrototypeEntry {
    fn from(entry: &PrototypeEntry) -> Self {
        Self {
            cluster: entry.cluster.clone(),
            digest: entry.digest.clone(),
            occurrence_count: entry.occurrence_count,
            doc_count: entry.doc_count,
            distinct_text_count: entry.distinct_text_count,
            max_byte_len: entry.max_byte_len,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PersistedShapeEntry {
    shape: String,
    digest: String,
    occurrence_count: usize,
    doc_count: usize,
    distinct_text_count: usize,
    max_byte_len: usize,
}

impl From<&ShapeEntry> for PersistedShapeEntry {
    fn from(entry: &ShapeEntry) -> Self {
        Self {
            shape: entry.shape.clone(),
            digest: entry.digest.clone(),
            occurrence_count: entry.occurrence_count,
            doc_count: entry.doc_count,
            distinct_text_count: entry.distinct_text_count,
            max_byte_len: entry.max_byte_len,
        }
    }
}

fn validate_persisted_vocab(persisted: &PersistedFaceoffVocab) -> Result<(), FractalError> {
    if persisted.version != FACEOFF_VOCAB_FORMAT_VERSION {
        return Err(FractalError::InvalidState(format!(
            "unsupported faceoff vocab version {} (expected {})",
            persisted.version, FACEOFF_VOCAB_FORMAT_VERSION
        )));
    }
    if persisted
        .motifs
        .windows(2)
        .any(|pair| pair[0].text >= pair[1].text)
    {
        return Err(FractalError::InvalidState(
            "persisted faceoff vocab motifs must be sorted and unique by text".to_string(),
        ));
    }
    if persisted
        .motifs
        .iter()
        .any(|entry| entry.digests.is_empty())
    {
        return Err(FractalError::InvalidState(
            "persisted faceoff vocab entries must include at least one digest".to_string(),
        ));
    }
    if persisted
        .motifs
        .iter()
        .any(|entry| entry.digests.windows(2).any(|pair| pair[0] >= pair[1]))
    {
        return Err(FractalError::InvalidState(
            "persisted faceoff vocab digest aliases must be sorted and unique".to_string(),
        ));
    }
    if persisted
        .prototypes
        .windows(2)
        .any(|pair| pair[0].cluster >= pair[1].cluster)
    {
        return Err(FractalError::InvalidState(
            "persisted faceoff prototype motifs must be sorted and unique by cluster".to_string(),
        ));
    }
    if persisted
        .shapes
        .windows(2)
        .any(|pair| pair[0].shape >= pair[1].shape)
    {
        return Err(FractalError::InvalidState(
            "persisted faceoff shape motifs must be sorted and unique by shape".to_string(),
        ));
    }
    if persisted.byte_fallback_base
        != (persisted.motifs.len() + persisted.prototypes.len() + persisted.shapes.len()) as u32
    {
        return Err(FractalError::InvalidState(format!(
            "persisted byte fallback base {} does not match motif count {}",
            persisted.byte_fallback_base,
            persisted.motifs.len() + persisted.prototypes.len() + persisted.shapes.len()
        )));
    }
    Ok(())
}

#[derive(Default)]
struct MotifAggregate {
    occurrence_count: usize,
    doc_ids: BTreeSet<usize>,
    digests: BTreeSet<String>,
    min_depth: usize,
    max_depth: usize,
    max_byte_len: usize,
    seen_once: bool,
}

impl MotifAggregate {
    fn observe(&mut self, record: &TokenRecord, doc_index: usize) -> Result<(), FractalError> {
        self.occurrence_count += 1;
        self.doc_ids.insert(doc_index);
        self.digests.insert(token_digest(&record.token)?.to_owned());
        if !self.seen_once {
            self.min_depth = record.depth;
            self.max_depth = record.depth;
            self.max_byte_len = record.end.saturating_sub(record.start);
            self.seen_once = true;
            return Ok(());
        }

        self.min_depth = self.min_depth.min(record.depth);
        self.max_depth = self.max_depth.max(record.depth);
        self.max_byte_len = self
            .max_byte_len
            .max(record.end.saturating_sub(record.start));
        Ok(())
    }

    fn into_entry(self, text: String, config: FaceoffVocabConfig) -> Option<VocabEntry> {
        let doc_count = self.doc_ids.len();
        if self.occurrence_count < config.min_occurrence_count || doc_count < config.min_doc_count {
            return None;
        }
        if config
            .max_token_bytes
            .is_some_and(|limit| self.max_byte_len > limit)
        {
            return None;
        }

        let digests = self.digests.into_iter().collect::<Vec<_>>();
        let digest = digests.first()?.clone();
        Some(VocabEntry {
            id: FaceoffTokenId::new(0),
            digest,
            text,
            digests,
            occurrence_count: self.occurrence_count,
            doc_count,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
            max_byte_len: self.max_byte_len,
        })
    }
}

#[derive(Default)]
struct ShapeAggregate {
    occurrence_count: usize,
    doc_ids: BTreeSet<usize>,
    distinct_texts: BTreeSet<String>,
    max_byte_len: usize,
}

#[derive(Default)]
struct PrototypeAggregate {
    occurrence_count: usize,
    doc_ids: BTreeSet<usize>,
    distinct_texts: BTreeSet<String>,
    max_byte_len: usize,
}

impl PrototypeAggregate {
    fn observe(
        &mut self,
        record: &TokenRecord,
        text: &str,
        doc_index: usize,
    ) -> Result<(), FractalError> {
        self.occurrence_count += 1;
        self.doc_ids.insert(doc_index);
        self.distinct_texts.insert(text.to_owned());
        self.max_byte_len = self
            .max_byte_len
            .max(record.end.saturating_sub(record.start));
        token_digest(&record.token)?;
        Ok(())
    }

    fn into_entry(self, cluster: String, config: FaceoffVocabConfig) -> Option<PrototypeEntry> {
        let doc_count = self.doc_ids.len();
        let distinct_text_count = self.distinct_texts.len();
        if self.occurrence_count < config.min_shape_occurrence_count
            || doc_count < config.min_shape_doc_count
            || distinct_text_count < config.min_shape_distinct_text_count
        {
            return None;
        }
        if config
            .max_token_bytes
            .is_some_and(|limit| self.max_byte_len > limit)
        {
            return None;
        }

        Some(PrototypeEntry {
            id: FaceoffTokenId::new(0),
            digest: prototype_digest(&cluster),
            cluster,
            occurrence_count: self.occurrence_count,
            doc_count,
            distinct_text_count,
            max_byte_len: self.max_byte_len,
        })
    }
}

impl ShapeAggregate {
    fn observe(
        &mut self,
        record: &TokenRecord,
        text: &str,
        doc_index: usize,
    ) -> Result<(), FractalError> {
        self.occurrence_count += 1;
        self.doc_ids.insert(doc_index);
        self.distinct_texts.insert(text.to_owned());
        self.max_byte_len = self
            .max_byte_len
            .max(record.end.saturating_sub(record.start));
        // Keep the same failure surface as exact aggregates if token formatting is invalid.
        token_digest(&record.token)?;
        Ok(())
    }

    fn into_entry(self, shape: String, config: FaceoffVocabConfig) -> Option<ShapeEntry> {
        let doc_count = self.doc_ids.len();
        let distinct_text_count = self.distinct_texts.len();
        if self.occurrence_count < config.min_shape_occurrence_count
            || doc_count < config.min_shape_doc_count
            || distinct_text_count < config.min_shape_distinct_text_count
        {
            return None;
        }
        if config
            .max_token_bytes
            .is_some_and(|limit| self.max_byte_len > limit)
        {
            return None;
        }

        Some(ShapeEntry {
            id: FaceoffTokenId::new(0),
            digest: format!("shape::{shape}"),
            shape,
            occurrence_count: self.occurrence_count,
            doc_count,
            distinct_text_count,
            max_byte_len: self.max_byte_len,
        })
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

fn prototype_digest(cluster: &str) -> String {
    let mut digest = 1469598103934665603u64;
    for byte in cluster.as_bytes() {
        digest ^= u64::from(*byte);
        digest = digest.wrapping_mul(1099511628211);
    }
    format!("prototype::{digest:016x}")
}

pub(crate) fn prototype_cluster_key(record: &TokenRecord) -> Result<String, FractalError> {
    let (prefix, _) = record
        .token
        .rsplit_once('-')
        .ok_or_else(|| {
            FractalError::InvalidState(format!(
                "token `{}` is missing digest suffix in expected dX-nY-qZ-digest format",
                record.token
            ))
        })?;
    let mut depth = None;
    let mut state_bin = None;
    for part in prefix.split('-') {
        if let Some(value) = part.strip_prefix('d') {
            depth = Some(value.parse::<usize>().map_err(|source| {
                FractalError::InvalidState(format!(
                    "failed to parse depth from token `{}`: {source}",
                    record.token
                ))
            })?);
        } else if let Some(value) = part.strip_prefix('q') {
            state_bin = Some(value.parse::<u16>().map_err(|source| {
                FractalError::InvalidState(format!(
                    "failed to parse state bin from token `{}`: {source}",
                    record.token
                ))
            })?);
        }
    }

    let depth = depth.ok_or_else(|| {
        FractalError::InvalidState(format!(
            "token `{}` is missing depth prefix for prototype clustering",
            record.token
        ))
    })?;
    let state_bin = state_bin.ok_or_else(|| {
        FractalError::InvalidState(format!(
            "token `{}` is missing state bin prefix for prototype clustering",
            record.token
        ))
    })?;
    let byte_len = record.end.saturating_sub(record.start);
    let len_bucket = byte_len_bucket(byte_len);
    let state_bucket = state_bin_bucket(state_bin);
    let shape = lexical_shape_key(&record.text);
    Ok(format!("d{depth}-lb{len_bucket}-qb{state_bucket}\u{001f}{shape}"))
}

fn byte_len_bucket(byte_len: usize) -> usize {
    if byte_len <= 1 {
        return byte_len;
    }
    usize::BITS as usize - (byte_len - 1).leading_zeros() as usize
}

fn state_bin_bucket(state_bin: u16) -> u16 {
    state_bin / 16
}
