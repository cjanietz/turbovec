//! UniFFI scaffolding for the Go bindings.
//!
//! The public Go API lives in the sibling `.go` files. This crate is the
//! `Send + Sync` interior-mutability wrapper UniFFI requires, mirroring
//! the Python binding's `RwLock` so concurrent searches stay concurrent.

uniffi::setup_scaffolding!();

use std::sync::{Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use turbovec_core::io::Durability;
use turbovec_core::{IdMapIndex as CoreIdMap, TurboQuantIndex as CoreIndex};

/// User-facing error. Flattened to its `Display` text so Go callers get
/// the same messages as Rust/`Display` (and Python `ValueError`).
#[derive(Debug, Clone, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum IndexError {
    #[error("{0}")]
    Message(String),
}

impl IndexError {
    fn msg(s: impl ToString) -> Self {
        Self::Message(s.to_string())
    }
}

impl From<turbovec_core::ConstructError> for IndexError {
    fn from(e: turbovec_core::ConstructError) -> Self {
        Self::msg(e)
    }
}

impl From<turbovec_core::AddError> for IndexError {
    fn from(e: turbovec_core::AddError) -> Self {
        Self::msg(e)
    }
}

impl From<turbovec_core::SearchError> for IndexError {
    fn from(e: turbovec_core::SearchError) -> Self {
        Self::msg(e)
    }
}

impl From<turbovec_core::CalibrateError> for IndexError {
    fn from(e: turbovec_core::CalibrateError) -> Self {
        Self::msg(e)
    }
}

fn io_err(path: &str, e: std::io::Error) -> IndexError {
    IndexError::msg(format!("{path}: {e}"))
}

fn io_err_bytes(e: std::io::Error) -> IndexError {
    IndexError::msg(e)
}

fn usize_of(name: &str, v: u64) -> Result<usize, IndexError> {
    usize::try_from(v).map_err(|_| IndexError::msg(format!("{name} {v} does not fit usize")))
}

/// Decode a little-endian float32 payload. UniFFI `bytes` is one copy of
/// packed floats; `sequence<f32>` would walk each element through a
/// `RustBuffer`.
fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>, IndexError> {
    if bytes.len() % 4 != 0 {
        return Err(IndexError::msg(format!(
            "float32 payload length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn lock_read<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(PoisonError::into_inner)
}

fn lock_write<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(PoisonError::into_inner)
}

fn durability(durable: bool) -> Durability {
    if durable {
        Durability::Durable
    } else {
        Durability::Fast
    }
}

fn calibration_name(state: turbovec_core::CalibrationState) -> String {
    match state {
        turbovec_core::CalibrationState::Uncalibrated => "uncalibrated".to_string(),
        turbovec_core::CalibrationState::Calibrated => "calibrated".to_string(),
        _ => unreachable!("unhandled CalibrationState variant"),
    }
}

fn empty_search() -> SearchResults {
    SearchResults {
        scores: Vec::new(),
        indices: Vec::new(),
        nq: 0,
        k: 0,
    }
}

fn empty_id_search() -> IdSearchResults {
    IdSearchResults {
        scores: Vec::new(),
        ids: Vec::new(),
        nq: 0,
        k: 0,
    }
}

/// Require `dim` to match a committed index dim. A lazy index has no dim
/// to disagree with; callers that still pass one are asking to split a
/// buffer that will not be searched.
fn check_search_dim(index_dim: Option<usize>, dim: u64) -> Result<Option<usize>, IndexError> {
    let Some(index_dim) = index_dim else {
        return Ok(None);
    };
    let dim = usize_of("dim", dim)?;
    if dim != index_dim {
        return Err(IndexError::msg(format!(
            "dim mismatch: index dim={index_dim}, query dim={dim}"
        )));
    }
    Ok(Some(index_dim))
}

#[derive(uniffi::Record)]
pub struct SearchResults {
    pub scores: Vec<f32>,
    pub indices: Vec<i64>,
    pub nq: u64,
    pub k: u64,
}

impl From<turbovec_core::SearchResults> for SearchResults {
    fn from(r: turbovec_core::SearchResults) -> Self {
        Self {
            scores: r.scores,
            indices: r.indices,
            nq: r.nq as u64,
            k: r.k as u64,
        }
    }
}

#[derive(uniffi::Record)]
pub struct IdSearchResults {
    pub scores: Vec<f32>,
    pub ids: Vec<u64>,
    pub nq: u64,
    pub k: u64,
}

impl From<turbovec_core::IdSearchResults> for IdSearchResults {
    fn from(r: turbovec_core::IdSearchResults) -> Self {
        Self {
            scores: r.scores,
            ids: r.ids,
            nq: r.nq as u64,
            k: r.k as u64,
        }
    }
}

#[uniffi::export(with_foreign)]
pub trait WarningHandler: Send + Sync {
    fn on_warning(&self, message: String);
}

static GO_HANDLER: Mutex<Option<Arc<dyn WarningHandler>>> = Mutex::new(None);

fn rust_warning_hook(message: &str) {
    let handler = GO_HANDLER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if let Some(h) = handler {
        h.on_warning(message.to_string());
    }
}

#[uniffi::export]
fn set_warning_handler(handler: Option<Arc<dyn WarningHandler>>) {
    let mut slot = GO_HANDLER.lock().unwrap_or_else(PoisonError::into_inner);
    *slot = handler;
    if slot.is_some() {
        turbovec_core::set_warning_hook(Some(rust_warning_hook));
    } else {
        turbovec_core::set_warning_hook(None);
    }
}

#[uniffi::export]
fn max_dim() -> u64 {
    turbovec_core::MAX_DIM as u64
}

#[derive(uniffi::Object)]
pub struct TurboQuantIndex {
    inner: RwLock<CoreIndex>,
}

#[uniffi::export]
impl TurboQuantIndex {
    #[uniffi::constructor]
    pub fn new(dim: u64, bit_width: u64) -> Result<Arc<Self>, IndexError> {
        let dim = usize_of("dim", dim)?;
        let bit_width = usize_of("bit_width", bit_width)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(CoreIndex::new(dim, bit_width)?),
        }))
    }

    #[uniffi::constructor]
    pub fn new_lazy(bit_width: u64) -> Result<Arc<Self>, IndexError> {
        let bit_width = usize_of("bit_width", bit_width)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(CoreIndex::new_lazy(bit_width)?),
        }))
    }

    #[uniffi::constructor]
    pub fn load(path: String) -> Result<Arc<Self>, IndexError> {
        let idx = CoreIndex::load(&path).map_err(|e| io_err(&path, e))?;
        Ok(Arc::new(Self {
            inner: RwLock::new(idx),
        }))
    }

    #[uniffi::constructor]
    pub fn from_bytes(data: Vec<u8>) -> Result<Arc<Self>, IndexError> {
        let idx = CoreIndex::from_bytes(&data).map_err(io_err_bytes)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(idx),
        }))
    }

    pub fn add(&self, vectors: Vec<u8>, dim: u64) -> Result<(), IndexError> {
        let vectors = decode_f32_le(&vectors)?;
        let dim = usize_of("dim", dim)?;
        lock_write(&self.inner).add_2d(&vectors, dim)?;
        Ok(())
    }

    pub fn search(&self, queries: Vec<u8>, dim: u64, k: u64) -> Result<SearchResults, IndexError> {
        self.search_with_mask(queries, dim, k, None)
    }

    pub fn search_with_mask(
        &self,
        queries: Vec<u8>,
        dim: u64,
        k: u64,
        mask: Option<Vec<bool>>,
    ) -> Result<SearchResults, IndexError> {
        let queries = decode_f32_le(&queries)?;
        let k = usize_of("k", k)?;
        let guard = lock_read(&self.inner);
        match check_search_dim(guard.dim_opt(), dim)? {
            None => Ok(empty_search()),
            Some(_) => Ok(guard
                .try_search_with_mask(&queries, k, mask.as_deref())?
                .into()),
        }
    }

    pub fn calibrate(&self, sample: Vec<u8>, dim: u64) -> Result<(), IndexError> {
        let sample = decode_f32_le(&sample)?;
        let dim = usize_of("dim", dim)?;
        lock_write(&self.inner).calibrate_2d(&sample, dim)?;
        Ok(())
    }

    pub fn swap_remove(&self, idx: u64) -> Result<u64, IndexError> {
        let idx = usize_of("idx", idx)?;
        let mut guard = lock_write(&self.inner);
        if idx >= guard.len() {
            return Err(IndexError::msg(format!(
                "index {idx} out of bounds (n_vectors = {})",
                guard.len()
            )));
        }
        Ok(guard.swap_remove(idx) as u64)
    }

    pub fn prepare(&self) {
        lock_read(&self.inner).prepare();
    }

    pub fn write(&self, path: String) -> Result<(), IndexError> {
        self.write_with_durability(path, true)
    }

    pub fn write_with_durability(&self, path: String, durable: bool) -> Result<(), IndexError> {
        lock_read(&self.inner)
            .write_with_durability(&path, durability(durable))
            .map_err(|e| io_err(&path, e))
    }

    pub fn sync(&self, path: String) -> Result<(), IndexError> {
        lock_write(&self.inner)
            .sync(&path)
            .map_err(|e| io_err(&path, e))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        lock_read(&self.inner).to_bytes()
    }

    pub fn len(&self) -> u64 {
        lock_read(&self.inner).len() as u64
    }

    pub fn dim(&self) -> Option<u64> {
        lock_read(&self.inner).dim_opt().map(|d| d as u64)
    }

    pub fn bit_width(&self) -> u64 {
        lock_read(&self.inner).bit_width() as u64
    }

    pub fn calibration_state(&self) -> String {
        calibration_name(lock_read(&self.inner).calibration_state())
    }
}

#[derive(uniffi::Object)]
pub struct IdMapIndex {
    inner: RwLock<CoreIdMap>,
}

#[uniffi::export]
impl IdMapIndex {
    #[uniffi::constructor]
    pub fn new(dim: u64, bit_width: u64) -> Result<Arc<Self>, IndexError> {
        let dim = usize_of("dim", dim)?;
        let bit_width = usize_of("bit_width", bit_width)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(CoreIdMap::new(dim, bit_width)?),
        }))
    }

    #[uniffi::constructor]
    pub fn new_lazy(bit_width: u64) -> Result<Arc<Self>, IndexError> {
        let bit_width = usize_of("bit_width", bit_width)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(CoreIdMap::new_lazy(bit_width)?),
        }))
    }

    #[uniffi::constructor]
    pub fn load(path: String) -> Result<Arc<Self>, IndexError> {
        let idx = CoreIdMap::load(&path).map_err(|e| io_err(&path, e))?;
        Ok(Arc::new(Self {
            inner: RwLock::new(idx),
        }))
    }

    #[uniffi::constructor]
    pub fn from_bytes(data: Vec<u8>) -> Result<Arc<Self>, IndexError> {
        let idx = CoreIdMap::from_bytes(&data).map_err(io_err_bytes)?;
        Ok(Arc::new(Self {
            inner: RwLock::new(idx),
        }))
    }

    pub fn add_with_ids(
        &self,
        vectors: Vec<u8>,
        dim: u64,
        ids: Vec<u64>,
    ) -> Result<(), IndexError> {
        let vectors = decode_f32_le(&vectors)?;
        let dim = usize_of("dim", dim)?;
        lock_write(&self.inner).add_with_ids_2d(&vectors, dim, &ids)?;
        Ok(())
    }

    pub fn remove(&self, id: u64) -> bool {
        lock_write(&self.inner).remove(id)
    }

    pub fn contains(&self, id: u64) -> bool {
        lock_read(&self.inner).contains(id)
    }

    pub fn search(
        &self,
        queries: Vec<u8>,
        dim: u64,
        k: u64,
    ) -> Result<IdSearchResults, IndexError> {
        self.search_with_allowlist(queries, dim, k, None)
    }

    pub fn search_with_allowlist(
        &self,
        queries: Vec<u8>,
        dim: u64,
        k: u64,
        allowlist: Option<Vec<u64>>,
    ) -> Result<IdSearchResults, IndexError> {
        let queries = decode_f32_le(&queries)?;
        let k = usize_of("k", k)?;
        let guard = lock_read(&self.inner);
        match check_search_dim(guard.dim_opt(), dim)? {
            None => Ok(empty_id_search()),
            Some(_) => Ok(guard
                .try_search_with_allowlist(&queries, k, allowlist.as_deref())?
                .into()),
        }
    }

    pub fn calibrate(&self, sample: Vec<u8>, dim: u64) -> Result<(), IndexError> {
        let sample = decode_f32_le(&sample)?;
        let dim = usize_of("dim", dim)?;
        lock_write(&self.inner).calibrate_2d(&sample, dim)?;
        Ok(())
    }

    pub fn prepare(&self) {
        lock_read(&self.inner).prepare();
    }

    pub fn write(&self, path: String) -> Result<(), IndexError> {
        self.write_with_durability(path, true)
    }

    pub fn write_with_durability(&self, path: String, durable: bool) -> Result<(), IndexError> {
        lock_read(&self.inner)
            .write_with_durability(&path, durability(durable))
            .map_err(|e| io_err(&path, e))
    }

    pub fn sync(&self, path: String) -> Result<(), IndexError> {
        lock_write(&self.inner)
            .sync(&path)
            .map_err(|e| io_err(&path, e))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        lock_read(&self.inner).to_bytes()
    }

    pub fn len(&self) -> u64 {
        lock_read(&self.inner).len() as u64
    }

    pub fn dim(&self) -> Option<u64> {
        lock_read(&self.inner).dim_opt().map(|d| d as u64)
    }

    pub fn bit_width(&self) -> u64 {
        lock_read(&self.inner).bit_width() as u64
    }

    pub fn calibration_state(&self) -> String {
        calibration_name(lock_read(&self.inner).calibration_state())
    }
}

#[cfg(test)]
mod tests {
    use super::decode_f32_le;

    #[test]
    fn decode_rejects_ragged_payload() {
        assert!(decode_f32_le(&[0, 0, 0]).is_err());
    }

    #[test]
    fn decode_round_trips_le_floats() {
        let mut bytes = Vec::new();
        for v in [1.0f32, -2.5, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let got = decode_f32_le(&bytes).unwrap();
        assert_eq!(got, vec![1.0, -2.5, 0.0]);
    }
}
