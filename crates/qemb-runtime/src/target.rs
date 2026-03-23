// GPU target support - implemented in Task 2

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Gfx1100,
    Gfx1103,
}

impl GpuTarget {
    pub fn mcpu_str(&self) -> &'static str {
        match self {
            GpuTarget::Gfx1100 => "gfx1100",
            GpuTarget::Gfx1103 => "gfx1103",
        }
    }

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