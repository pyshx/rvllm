//! Persistent cache for tuned kernel configurations.
//!
//! Stores winning (kernel, shape, GPU) -> config mappings on disk
//! so subsequent runs skip the tuning phase.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Cache key: identifies a specific kernel + shape + GPU combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TuneCacheKey {
    /// Kernel function name.
    pub kernel_name: String,
    /// GPU model name (e.g., "NVIDIA H100 80GB HBM3").
    pub gpu_name: String,
    /// Problem shape parameters that affect tuning.
    pub shape: Vec<u32>,
}

impl TuneCacheKey {
    pub fn new(kernel_name: &str, gpu_name: &str, shape: &[u32]) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            gpu_name: gpu_name.to_string(),
            shape: shape.to_vec(),
        }
    }
}

/// A tuned kernel configuration to persist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunedConfig {
    /// Optimal block dimensions.
    pub block: (u32, u32, u32),
    /// Optimal shared memory size.
    pub shared_mem: u32,
    /// Extra tuning parameters.
    pub extra: Vec<(String, u32)>,
    /// Measured average time in nanoseconds.
    pub avg_ns: u64,
    /// Speedup vs default config.
    pub speedup: f64,
    /// When this was tuned (unix timestamp).
    pub tuned_at: u64,
}

/// Persistent disk cache for tuned configs.
pub struct TuneCache {
    path: PathBuf,
    entries: HashMap<TuneCacheKey, TunedConfig>,
}

impl TuneCache {
    /// Load or create a tune cache at the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let entries = Self::load_from_disk(&path).unwrap_or_default();
        if !entries.is_empty() {
            info!(count = entries.len(), path = %path.display(), "loaded tune cache");
        }
        Self { path, entries }
    }

    /// Default cache path: `$RVLLM_TUNE_CACHE` or `~/.cache/rvllm/tune.json`.
    pub fn default_path() -> PathBuf {
        if let Ok(p) = std::env::var("RVLLM_TUNE_CACHE") {
            return PathBuf::from(p);
        }
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(home).join(".cache/rvllm/tune.json")
    }

    /// Look up a tuned config.
    pub fn get(&self, key: &TuneCacheKey) -> Option<&TunedConfig> {
        self.entries.get(key)
    }

    /// Insert a tuned config and save to disk.
    pub fn insert(&mut self, key: TuneCacheKey, config: TunedConfig) {
        debug!(kernel = %key.kernel_name, speedup = config.speedup, "caching tuned config");
        self.entries.insert(key, config);
        if let Err(e) = self.save_to_disk() {
            warn!("failed to save tune cache: {e}");
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Check if a key is cached.
    pub fn contains(&self, key: &TuneCacheKey) -> bool {
        self.entries.contains_key(key)
    }

    fn load_from_disk(path: &Path) -> Option<HashMap<TuneCacheKey, TunedConfig>> {
        let content = std::fs::read_to_string(path).ok()?;
        let entries: Vec<(TuneCacheKey, TunedConfig)> = serde_json::from_str(&content).ok()?;
        Some(entries.into_iter().collect())
    }

    fn save_to_disk(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let entries: Vec<(&TuneCacheKey, &TunedConfig)> = self.entries.iter().collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&self.path, json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tune.json");

        let mut cache = TuneCache::new(&path);
        assert!(cache.is_empty());

        let key = TuneCacheKey::new("my_kernel", "H100", &[32, 4608, 3584]);
        let config = TunedConfig {
            block: (256, 1, 1),
            shared_mem: 32768,
            extra: vec![("tile_k".into(), 64)],
            avg_ns: 11000,
            speedup: 1.15,
            tuned_at: 1234567890,
        };

        cache.insert(key.clone(), config.clone());
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&key));

        // Reload from disk
        let cache2 = TuneCache::new(&path);
        assert_eq!(cache2.len(), 1);
        let loaded = cache2.get(&key).unwrap();
        assert_eq!(loaded.block, (256, 1, 1));
        assert_eq!(loaded.speedup, 1.15);
    }

    #[test]
    fn cache_missing_file() {
        let cache = TuneCache::new("/nonexistent/tune.json");
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_key_equality() {
        let k1 = TuneCacheKey::new("kern", "H100", &[1, 2, 3]);
        let k2 = TuneCacheKey::new("kern", "H100", &[1, 2, 3]);
        let k3 = TuneCacheKey::new("kern", "A100", &[1, 2, 3]);
        assert_eq!(k1, k2);
        assert_ne!(k1, k3);
    }
}
