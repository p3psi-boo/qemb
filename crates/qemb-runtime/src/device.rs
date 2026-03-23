//! GPU device detection via KFD topology
//!
//! This module provides device detection and information gathering
//! for AMD GPUs through the KFD (Kernel Fusion Driver) sysfs interface.

use crate::target::GpuTarget;
use qemb_common::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Represents an AMD GPU device
#[derive(Debug, Clone)]
pub struct Device {
    /// GPU target architecture
    pub target: GpuTarget,
    /// Path to KFD device (e.g., /dev/kfd)
    pub kfd_path: PathBuf,
    /// Path to render node (e.g., /dev/dri/renderD128)
    pub render_node: PathBuf,
    /// Device name from sysfs
    pub name: String,
}

impl Device {
    /// Probe for available GPU devices via KFD
    ///
    /// Uses default paths for KFD device and render node.
    pub fn probe() -> Result<Self> {
        Self::probe_paths(
            Path::new("/dev/kfd"),
            Path::new("/dev/dri/renderD128"),
        )
    }

    /// Probe with explicit paths
    ///
    /// # Arguments
    /// * `kfd_path` - Path to KFD device
    /// * `render_node` - Path to render node
    pub fn probe_paths(kfd_path: &Path, render_node: &Path) -> Result<Self> {
        // Check KFD device exists
        if !kfd_path.exists() {
            return Err(Error::Gpu(format!(
                "KFD device not found: {:?}",
                kfd_path
            )));
        }

        // Check render node exists
        if !render_node.exists() {
            return Err(Error::Gpu(format!(
                "Render node not found: {:?}",
                render_node
            )));
        }

        // Detect GPU target from sysfs
        let target = Self::detect_target()?;

        // Get device name
        let name = Self::read_device_name().unwrap_or_else(|_| "unknown".to_string());

        Ok(Device {
            target,
            kfd_path: kfd_path.to_path_buf(),
            render_node: render_node.to_path_buf(),
            name,
        })
    }

    /// Detect GPU target from sysfs
    fn detect_target() -> Result<GpuTarget> {
        // Try to read from KFD topology
        let gfx_version = Self::read_gfx_version()?;
        GpuTarget::from_str(&gfx_version)
            .ok_or_else(|| Error::Gpu(format!("Unknown GPU target: {}", gfx_version)))
    }

    /// Read GFX version from sysfs
    ///
    /// Looks through KFD topology nodes to find the GPU's GFX version.
    fn read_gfx_version() -> Result<String> {
        let topology_path = Path::new("/sys/class/kfd/kfd/topology/nodes");

        if !topology_path.exists() {
            return Err(Error::Gpu("KFD topology not found".to_string()));
        }

        // Iterate through nodes to find GPU
        for entry in fs::read_dir(topology_path).map_err(|e| {
            Error::Gpu(format!("Failed to read KFD topology: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                Error::Gpu(format!("Failed to read topology entry: {}", e))
            })?;

            let node_path = entry.path();
            let gfx_version_path = node_path.join("gfx_version");

            if gfx_version_path.exists() {
                if let Ok(version) = fs::read_to_string(&gfx_version_path) {
                    let version = version.trim();
                    // Check if it's a valid RDNA3 target (version starts with "11" for gfx11xx)
                    if version.starts_with("11") {
                        // Format is like "11003" for gfx1103
                        return Ok(format!("gfx{}", version));
                    }
                }
            }
        }

        Err(Error::Gpu("No GPU found in KFD topology".to_string()))
    }

    /// Read device name from sysfs
    fn read_device_name() -> Result<String> {
        let topology_path = Path::new("/sys/class/kfd/kfd/topology/nodes");

        for entry in fs::read_dir(topology_path).map_err(|e| {
            Error::Gpu(format!("Failed to read KFD topology: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                Error::Gpu(format!("Failed to read topology entry: {}", e))
            })?;

            let node_path = entry.path();
            let name_path = node_path.join("name");

            if name_path.exists() {
                if let Ok(name) = fs::read_to_string(&name_path) {
                    let name = name.trim().to_string();
                    // Skip CPU nodes (name is "cpu")
                    if !name.is_empty() && name != "cpu" {
                        return Ok(name);
                    }
                }
            }
        }

        Err(Error::Gpu("No GPU name found".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_requires_kfd() {
        // This test will fail if /dev/kfd doesn't exist
        // which is expected in non-GPU environments
        let result = Device::probe();
        // Just check that it doesn't panic and returns a Result
        match result {
            Ok(device) => {
                // If we have a GPU, check the target is valid
                assert!(matches!(device.target, GpuTarget::Gfx1100 | GpuTarget::Gfx1103));
            }
            Err(e) => {
                // If no GPU, we should get an error
                // This is fine for CI environments
                println!("Probe result (expected in non-GPU env): {}", e);
            }
        }
    }

    #[test]
    fn test_probe_paths_nonexistent() {
        let result = Device::probe_paths(
            Path::new("/nonexistent/kfd"),
            Path::new("/nonexistent/renderD128"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_device_fields() {
        // Test that Device struct can be created with expected fields
        let device = Device {
            target: GpuTarget::Gfx1103,
            kfd_path: PathBuf::from("/dev/kfd"),
            render_node: PathBuf::from("/dev/dri/renderD128"),
            name: "AMD Radeon 780M".to_string(),
        };

        assert_eq!(device.target, GpuTarget::Gfx1103);
        assert_eq!(device.kfd_path, PathBuf::from("/dev/kfd"));
        assert_eq!(device.render_node, PathBuf::from("/dev/dri/renderD128"));
        assert_eq!(device.name, "AMD Radeon 780M");
    }
}