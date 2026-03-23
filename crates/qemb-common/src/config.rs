use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub model_path: PathBuf,
    pub max_seq_len: usize,
    pub device: DeviceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub kfd_path: PathBuf,
    pub render_node: PathBuf,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model"),
            max_seq_len: 512,
            device: DeviceConfig {
                kfd_path: PathBuf::from("/dev/kfd"),
                render_node: PathBuf::from("/dev/dri/renderD128"),
            },
        }
    }
}