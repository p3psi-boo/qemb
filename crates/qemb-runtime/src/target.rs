//! GPU target support for AMD RDNA3 architecture
//!
//! This module provides target identification and code generation support
//! for AMD RDNA3 GPUs including gfx1100 and gfx1103.

use std::fmt;

/// GPU target architecture for code generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    /// AMD RDNA3 gfx1100 (Radeon Pro W7900, etc.)
    Gfx1100,
    /// AMD RDNA3 gfx1103 (Radeon 780M iGPU)
    Gfx1103,
}

impl GpuTarget {
    /// Returns the LLVM target CPU string for code generation
    pub fn mcpu_str(&self) -> &'static str {
        match self {
            GpuTarget::Gfx1100 => "gfx1100",
            GpuTarget::Gfx1103 => "gfx1103",
        }
    }

    /// Returns the AMDGPU code object target triple suffix
    pub fn code_object_target(&self) -> &'static str {
        match self {
            GpuTarget::Gfx1100 => "amdgcn-amd-amdhsa--gfx1100",
            GpuTarget::Gfx1103 => "amdgcn-amd-amdhsa--gfx1103",
        }
    }

    /// Parse from string (e.g., from /sys/class/kfd/kfd/topology/nodes/*/simd_ids)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "gfx1100" => Some(GpuTarget::Gfx1100),
            "gfx1103" => Some(GpuTarget::Gfx1103),
            _ => None,
        }
    }
}

impl fmt::Display for GpuTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mcpu_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gfx1100_mcpu_str() {
        let target = GpuTarget::Gfx1100;
        assert_eq!(target.mcpu_str(), "gfx1100");
    }

    #[test]
    fn test_gfx1103_mcpu_str() {
        let target = GpuTarget::Gfx1103;
        assert_eq!(target.mcpu_str(), "gfx1103");
    }

    #[test]
    fn test_gfx1100_code_object_target() {
        let target = GpuTarget::Gfx1100;
        assert_eq!(target.code_object_target(), "amdgcn-amd-amdhsa--gfx1100");
    }

    #[test]
    fn test_gfx1103_code_object_target() {
        let target = GpuTarget::Gfx1103;
        assert_eq!(target.code_object_target(), "amdgcn-amd-amdhsa--gfx1103");
    }

    #[test]
    fn test_from_str_gfx1100() {
        assert_eq!(GpuTarget::from_str("gfx1100"), Some(GpuTarget::Gfx1100));
        assert_eq!(GpuTarget::from_str("GFX1100"), Some(GpuTarget::Gfx1100));
    }

    #[test]
    fn test_from_str_gfx1103() {
        assert_eq!(GpuTarget::from_str("gfx1103"), Some(GpuTarget::Gfx1103));
        assert_eq!(GpuTarget::from_str("GFX1103"), Some(GpuTarget::Gfx1103));
    }

    #[test]
    fn test_from_str_unknown() {
        assert_eq!(GpuTarget::from_str("unknown"), None);
        assert_eq!(GpuTarget::from_str("gfx900"), None);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", GpuTarget::Gfx1100), "gfx1100");
        assert_eq!(format!("{}", GpuTarget::Gfx1103), "gfx1103");
    }
}