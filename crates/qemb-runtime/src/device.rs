// Device detection - implemented in Task 3

use crate::target::GpuTarget;
use qemb_common::{Error, Result};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Device {
    pub target: GpuTarget,
    pub kfd_path: PathBuf,
    pub render_node: PathBuf,
    pub name: String,
}

impl Device {
    pub fn probe() -> Result<Self> {
        Err(Error::Gpu("Device detection not implemented".to_string()))
    }
}