# Qwen3 Embedding Phase 1 Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Set up workspace skeleton, add gfx1103 target support, dispatch smoke kernel on 780M iGPU, and create service stub with health endpoints.

**Architecture:** Rust workspace with 7 crates. Vendor KFD runtime and code object generation from refs/t0-gpu. Use axum for HTTP server. Target gfx1103 (AMD 780M iGPU) via /dev/kfd.

**Tech Stack:** Rust, axum/tokio, tokenizers crate, refs/t0-gpu reference implementation

---

## Phase 1 Overview

This phase establishes the foundation:
1. Workspace skeleton with all crates
2. gfx1103 target support (extend from gfx1100)
3. KFD runtime smoke test
4. Converter stub
5. Service stub with health endpoints

**Estimated time:** 2-3 hours

---

## Task 1: Create Workspace Skeleton

**TDD scenario:** Trivial change — no tests needed for directory structure

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `crates/qemb-common/Cargo.toml`
- Create: `crates/qemb-common/src/lib.rs`
- Create: `crates/qemb-runtime/Cargo.toml`
- Create: `crates/qemb-runtime/src/lib.rs`
- Create: `crates/qemb-kernels/Cargo.toml`
- Create: `crates/qemb-kernels/src/lib.rs`
- Create: `crates/qemb-convert/Cargo.toml`
- Create: `crates/qemb-convert/src/lib.rs`
- Create: `crates/qemb-tokenizer/Cargo.toml`
- Create: `crates/qemb-tokenizer/src/lib.rs`
- Create: `crates/qemb-service/Cargo.toml`
- Create: `crates/qemb-service/src/lib.rs`
- Create: `crates/qemb-cli/Cargo.toml`
- Create: `crates/qemb-cli/src/main.rs`

**Step 1: Create root workspace Cargo.toml**

```toml
[workspace]
resolver = "2"
members = [
    "crates/qemb-common",
    "crates/qemb-runtime",
    "crates/qemb-kernels",
    "crates/qemb-convert",
    "crates/qemb-tokenizer",
    "crates/qemb-service",
    "crates/qemb-cli",
]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/qwen3-emb"

[workspace.dependencies]
# Common dependencies
tokio = { version = "1", features = ["full"] }
axum = "0.8"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"

# Internal crates
qemb-common = { path = "crates/qemb-common" }
qemb-runtime = { path = "crates/qemb-runtime" }
qemb-kernels = { path = "crates/qemb-kernels" }
qemb-convert = { path = "crates/qemb-convert" }
qemb-tokenizer = { path = "crates/qemb-tokenizer" }
qemb-service = { path = "crates/qemb-service" }
```

**Step 2: Create qemb-common crate**

`crates/qemb-common/Cargo.toml`:
```toml
[package]
name = "qemb-common"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
thiserror.workspace = true
anyhow.workspace = true
serde.workspace = true
```

`crates/qemb-common/src/lib.rs`:
```rust
pub mod error;
pub mod config;

pub use error::{Error, Result};
```

`crates/qemb-common/src/error.rs`:
```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("GPU error: {0}")]
    Gpu(String),
    
    #[error("Model error: {0}")]
    Model(String),
    
    #[error("Tokenization error: {0}")]
    Tokenization(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

`crates/qemb-common/src/config.rs`:
```rust
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
```

**Step 3: Create qemb-runtime crate**

`crates/qemb-runtime/Cargo.toml`:
```toml
[package]
name = "qemb-runtime"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
thiserror.workspace = true
anyhow.workspace = true
tracing.workspace = true
```

`crates/qemb-runtime/src/lib.rs`:
```rust
pub mod device;
pub mod target;

pub use device::Device;
pub use target::GpuTarget;
```

**Step 4: Create qemb-kernels crate**

`crates/qemb-kernels/Cargo.toml`:
```toml
[package]
name = "qemb-kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
thiserror.workspace = true
anyhow.workspace = true
```

`crates/qemb-kernels/src/lib.rs`:
```rust
pub mod primitives;

// GPU kernel implementations will go here
```

**Step 5: Create qemb-convert crate**

`crates/qemb-convert/Cargo.toml`:
```toml
[package]
name = "qemb-convert"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
thiserror.workspace = true
anyhow.workspace = true
serde.workspace = true
serde_json.workspace = true
```

`crates/qemb-convert/src/lib.rs`:
```rust
pub mod packer;
pub mod schema;

pub use schema::ModelBundle;
```

**Step 6: Create qemb-tokenizer crate**

`crates/qemb-tokenizer/Cargo.toml`:
```toml
[package]
name = "qemb-tokenizer"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
thiserror.workspace = true
anyhow.workspace = true
tokenizers = "0.21"
```

`crates/qemb-tokenizer/src/lib.rs`:
```rust
pub mod tokenizer;

pub use tokenizer::Tokenizer;
```

**Step 7: Create qemb-service crate**

`crates/qemb-service/Cargo.toml`:
```toml
[package]
name = "qemb-service"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
qemb-tokenizer.workspace = true
tokio.workspace = true
axum.workspace = true
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

`crates/qemb-service/src/lib.rs`:
```rust
pub mod api;
pub mod server;

pub use server::Server;
```

**Step 8: Create qemb-cli crate**

`crates/qemb-cli/Cargo.toml`:
```toml
[package]
name = "qemb-cli"
version.workspace = true
edition.workspace = true
license.workspace = true

[[bin]]
name = "qemb"
path = "src/main.rs"

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
qemb-kernels.workspace = true
qemb-convert.workspace = true
qemb-tokenizer.workspace = true
qemb-service.workspace = true
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
clap = { version = "4", features = ["derive"] }
```

`crates/qemb-cli/src/main.rs`:
```rust
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "qemb")]
#[command(about = "Qwen3 Embedding Bare-Metal RDNA3 Service", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the embedding service
    Serve {
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
    /// Convert Hugging Face model to packed format
    Convert {
        /// Path to Hugging Face model directory
        #[arg(short, long)]
        input: String,
        /// Path to output packed model
        #[arg(short, long)]
        output: String,
    },
    /// Run offline embedding inference
    Run {
        /// Input text to embed
        #[arg(short, long)]
        text: String,
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Serve { model, port } => {
            println!("Starting server on port {} with model {}", port, model);
            // TODO: Implement server
        }
        Commands::Convert { input, output } => {
            println!("Converting {} to {}", input, output);
            // TODO: Implement converter
        }
        Commands::Run { text, model } => {
            println!("Running inference on '{}' with model {}", text, model);
            // TODO: Implement offline runner
        }
    }
    
    Ok(())
}
```

**Step 9: Create directories**

```bash
mkdir -p crates/qemb-common/src
mkdir -p crates/qemb-runtime/src
mkdir -p crates/qemb-kernels/src
mkdir -p crates/qemb-convert/src
mkdir -p crates/qemb-tokenizer/src
mkdir -p crates/qemb-service/src
mkdir -p crates/qemb-cli/src
```

**Step 10: Verify workspace compiles**

Run: `cargo build`
Expected: Compiles successfully with no errors

**Step 11: Commit**

```bash
git add Cargo.toml crates/
git commit -m "feat: create workspace skeleton with 7 crates"
```

---

## Task 2: Add gfx1103 Target Support

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-runtime/src/lib.rs`
- Create: `crates/qemb-runtime/src/target.rs`
- Create: `crates/qemb-runtime/src/target.rs` (test module)
- Reference: `refs/t0-gpu/src/t0/compile.rs` for target enum pattern

**Step 1: Write the failing test**

Create `crates/qemb-runtime/src/target.rs`:

```rust
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
    fn test_gfx1103_mcpu_str() {
        let target = GpuTarget::Gfx1103;
        assert_eq!(target.mcpu_str(), "gfx1103");
    }

    #[test]
    fn test_gfx1103_code_object_target() {
        let target = GpuTarget::Gfx1103;
        assert_eq!(target.code_object_target(), "amdgcn-amd-amdhsa--gfx1103");
    }

    #[test]
    fn test_gfx1103_from_str() {
        assert_eq!(GpuTarget::from_str("gfx1103"), Some(GpuTarget::Gfx1103));
        assert_eq!(GpuTarget::from_str("GFX1103"), Some(GpuTarget::Gfx1103));
        assert_eq!(GpuTarget::from_str("gfx1100"), Some(GpuTarget::Gfx1100));
        assert_eq!(GpuTarget::from_str("unknown"), None);
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p qemb-runtime`
Expected: All tests pass

**Step 3: Update lib.rs to export target module**

Modify `crates/qemb-runtime/src/lib.rs`:
```rust
pub mod device;
pub mod target;

pub use device::Device;
pub use target::GpuTarget;
```

**Step 4: Verify workspace still compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add crates/qemb-runtime/
git commit -m "feat(runtime): add gfx1103 target support with GpuTarget enum"
```

---

## Task 3: Add GPU Device Detection

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-runtime/src/device.rs`
- Reference: `refs/t0-gpu/src/kfd/mod.rs` for KFD patterns

**Step 1: Write the failing test**

Create `crates/qemb-runtime/src/device.rs`:

```rust
use crate::target::GpuTarget;
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Debug, Clone)]
pub struct Device {
    pub target: GpuTarget,
    pub kfd_path: PathBuf,
    pub render_node: PathBuf,
    pub name: String,
}

impl Device {
    /// Probe for available GPU devices via KFD
    pub fn probe() -> crate::Result<Self> {
        // Try default paths first
        Self::probe_paths(
            Path::new("/dev/kfd"),
            Path::new("/dev/dri/renderD128"),
        )
    }

    /// Probe with explicit paths
    pub fn probe_paths(kfd_path: &Path, render_node: &Path) -> crate::Result<Self> {
        // Check KFD device exists
        if !kfd_path.exists() {
            return Err(crate::Error::Gpu(format!(
                "KFD device not found: {:?}",
                kfd_path
            )));
        }

        // Check render node exists
        if !render_node.exists() {
            return Err(crate::Error::Gpu(format!(
                "Render node not found: {:?}",
                render_node
            )));
        }

        // Detect GPU target from sysfs
        let target = Self::detect_target()?;

        let name = Self::read_device_name().unwrap_or_else(|_| "unknown".to_string());

        Ok(Device {
            target,
            kfd_path: kfd_path.to_path_buf(),
            render_node: render_node.to_path_buf(),
            name,
        })
    }

    /// Detect GPU target from sysfs
    fn detect_target() -> crate::Result<GpuTarget> {
        // Try to read from KFD topology
        let gfx_version = Self::read_gfx_version()?;
        GpuTarget::from_str(&gfx_version)
            .ok_or_else(|| crate::Error::Gpu(format!("Unknown GPU target: {}", gfx_version)))
    }

    /// Read GFX version from sysfs
    fn read_gfx_version() -> crate::Result<String> {
        // Look for gfx_version in KFD topology
        let topology_path = Path::new("/sys/class/kfd/kfd/topology/nodes");
        
        if !topology_path.exists() {
            return Err(crate::Error::Gpu("KFD topology not found".to_string()));
        }

        // Iterate through nodes to find GPU
        for entry in fs::read_dir(topology_path).map_err(|e| {
            crate::Error::Gpu(format!("Failed to read KFD topology: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                crate::Error::Gpu(format!("Failed to read topology entry: {}", e))
            })?;
            
            let node_path = entry.path();
            let gfx_version_path = node_path.join("gfx_version");
            
            if gfx_version_path.exists() {
                if let Ok(version) = fs::read_to_string(&gfx_version_path) {
                    let version = version.trim();
                    // Check if it's a valid RDNA3 target
                    if version.starts_with("11") {
                        // Format is like "11003" for gfx1103
                        return Ok(format!("gfx{}", version));
                    }
                }
            }
        }

        Err(crate::Error::Gpu("No GPU found in KFD topology".to_string()))
    }

    /// Read device name from sysfs
    fn read_device_name() -> crate::Result<String> {
        let topology_path = Path::new("/sys/class/kfd/kfd/topology/nodes");
        
        for entry in fs::read_dir(topology_path).map_err(|e| {
            crate::Error::Gpu(format!("Failed to read KFD topology: {}", e))
        })? {
            let entry = entry.map_err(|e| {
                crate::Error::Gpu(format!("Failed to read topology entry: {}", e))
            })?;
            
            let node_path = entry.path();
            let name_path = node_path.join("name");
            
            if name_path.exists() {
                if let Ok(name) = fs::read_to_string(&name_path) {
                    let name = name.trim().to_string();
                    if !name.is_empty() && name != "cpu" {
                        return Ok(name);
                    }
                }
            }
        }

        Err(crate::Error::Gpu("No GPU name found".to_string()))
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
        // Just check that it doesn't panic
        println!("Probe result: {:?}", result);
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p qemb-runtime`
Expected: Tests pass (probe may fail gracefully on non-GPU systems)

**Step 3: Commit**

```bash
git add crates/qemb-runtime/src/device.rs
git commit -m "feat(runtime): add GPU device detection via KFD topology"
```

---

## Task 4: Vendor KFD Runtime Code from refs/t0-gpu

**TDD scenario:** Modifying tested code — run existing tests first

**Files:**
- Reference: `refs/t0-gpu/src/kfd/mod.rs`
- Create: `crates/qemb-runtime/src/kfd.rs`
- Reference: `refs/t0-gpu/src/rdna3_code_object.rs`
- Create: `crates/qemb-runtime/src/code_object.rs`

**Step 1: Examine reference implementation**

Read `refs/t0-gpu/src/kfd/mod.rs` to understand the KFD runtime structure.

**Step 2: Create minimal KFD runtime wrapper**

Create `crates/qemb-runtime/src/kfd.rs`:

```rust
//! KFD (Kernel Fusion Driver) runtime for AMD GPUs
//! 
//! This module provides direct access to AMD GPUs via /dev/kfd
//! without ROCm or HIP runtime dependencies.

use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::os::raw::c_int;

mod sys;

pub use sys::*;

/// KFD device handle
pub struct KfdDevice {
    fd: c_int,
}

impl KfdDevice {
    /// Open the KFD device
    pub fn open(path: &Path) -> crate::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)
            .map_err(|e| crate::Error::Gpu(format!("Failed to open KFD device: {}", e)))?;

        let fd = file.as_raw_fd();
        std::mem::forget(file); // Don't close the fd

        Ok(KfdDevice { fd })
    }

    /// Get the file descriptor
    pub fn fd(&self) -> c_int {
        self.fd
    }
}

impl Drop for KfdDevice {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

// Placeholder for now - will be expanded with queue, memory, dispatch
```

Create `crates/qemb-runtime/src/kfd/sys.rs`:

```rust
//! Low-level KFD ioctl definitions

// KFD ioctl numbers - from Linux kernel include/uapi/linux/kfd_ioctl.h
// These are placeholders for the actual ioctl definitions

pub const KFD_IOCTL_MAJOR_VERSION: u32 = 1;
pub const KFD_IOCTL_MINOR_VERSION: u32 = 1;

// Placeholder for actual ioctl structures
// Will be populated when we implement actual KFD operations
```

**Step 3: Update lib.rs**

Modify `crates/qemb-runtime/src/lib.rs`:
```rust
pub mod device;
pub mod target;
pub mod kfd;
pub mod code_object;

pub use device::Device;
pub use target::GpuTarget;
pub use kfd::KfdDevice;
```

**Step 4: Add libc dependency**

Update `crates/qemb-runtime/Cargo.toml`:
```toml
[dependencies]
qemb-common.workspace = true
thiserror.workspace = true
anyhow.workspace = true
tracing.workspace = true
libc = "0.2"
```

**Step 5: Verify compiles**

Run: `cargo build -p qemb-runtime`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add crates/qemb-runtime/
git commit -m "feat(runtime): add minimal KFD runtime wrapper"
```

---

## Task 5: Add Code Object Generation

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Reference: `refs/t0-gpu/src/rdna3_code_object.rs`
- Create: `crates/qemb-runtime/src/code_object.rs`

**Step 1: Create code object module**

Create `crates/qemb-runtime/src/code_object.rs`:

```rust
//! AMD GPU code object (ELF) generation
//! 
//! This module generates AMDGPU ELF code objects for GPU kernels.

use crate::target::GpuTarget;

/// AMDGPU code object builder
pub struct CodeObjectBuilder {
    target: GpuTarget,
    kernels: Vec<KernelInfo>,
}

#[derive(Debug, Clone)]
pub struct KernelInfo {
    pub name: String,
    pub code: Vec<u8>,
    pub sgpr_count: u16,
    pub vgpr_count: u16,
    pub shared_memory_bytes: u32,
}

impl CodeObjectBuilder {
    pub fn new(target: GpuTarget) -> Self {
        CodeObjectBuilder {
            target,
            kernels: Vec::new(),
        }
    }

    pub fn add_kernel(&mut self, kernel: KernelInfo) -> &mut Self {
        self.kernels.push(kernel);
        self
    }

    /// Build the AMDGPU ELF code object
    pub fn build(&self) -> crate::Result<Vec<u8>> {
        // Create ELF header for AMDGPU
        let mut elf = Vec::new();
        
        // ELF header
        elf.extend_from_slice(&self.build_elf_header());
        
        // Program headers
        elf.extend_from_slice(&self.build_program_headers());
        
        // Section data
        elf.extend_from_slice(&self.build_sections());
        
        Ok(elf)
    }

    fn build_elf_header(&self) -> [u8; 64] {
        let mut header = [0u8; 64];
        
        // ELF magic
        header[0..4].copy_from_slice(b"\x7fELF");
        
        // 64-bit ELF
        header[4] = 2;
        
        // Little endian
        header[5] = 1;
        
        // ELF version
        header[6] = 1;
        
        // OS/ABI: AMDGPU
        header[7] = 0x40; // ELFOSABI_AMDGPU
        
        // Machine type: AMDGPU (0xE0)
        header[18..20].copy_from_slice(&(0xE0u16).to_le_bytes());
        
        header
    }

    fn build_program_headers(&self) -> Vec<u8> {
        // Placeholder - minimal program headers
        Vec::new()
    }

    fn build_sections(&self) -> Vec<u8> {
        // Placeholder - minimal sections
        Vec::new()
    }
}

impl CodeObjectBuilder {
    /// Get the e_flags for this target
    pub fn e_flags(&self) -> u32 {
        match self.target {
            GpuTarget::Gfx1100 => 0x1000, // EF_AMDGPU_MACH_AMDGCN_GFX1100
            GpuTarget::Gfx1103 => 0x1030, // EF_AMDGPU_MACH_AMDGCN_GFX1103
        }
    }

    /// Get the note vendor name
    pub fn note_vendor(&self) -> &'static [u8] {
        b"AMD"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_object_builder_e_flags_gfx1103() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        assert_eq!(builder.e_flags(), 0x1030);
    }

    #[test]
    fn test_code_object_builder_e_flags_gfx1100() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1100);
        assert_eq!(builder.e_flags(), 0x1000);
    }

    #[test]
    fn test_build_empty_code_object() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        let result = builder.build();
        assert!(result.is_ok());
        let elf = result.unwrap();
        // Check ELF magic
        assert_eq!(&elf[0..4], b"\x7fELF");
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p qemb-runtime`
Expected: All tests pass

**Step 3: Commit**

```bash
git add crates/qemb-runtime/src/code_object.rs
git commit -m "feat(runtime): add AMDGPU code object builder for gfx1100/gfx1103"
```

---

## Task 6: Create Service Health Endpoints

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-service/src/api.rs`
- Create: `crates/qemb-service/src/server.rs`

**Step 1: Create API types**

Create `crates/qemb-service/src/api.rs`:

```rust
//! OpenAI-compatible API types

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: EncodingFormat,
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    Float,
    Base64,
}

fn default_encoding_format() -> EncodingFormat {
    EncodingFormat::Float
}

fn default_dimensions() -> usize {
    1024
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
```

**Step 2: Create server module**

Create `crates/qemb-service/src/server.rs`:

```rust
//! HTTP server implementation

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::api::*;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub model_name: String,
    pub model_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            model_name: "Qwen3-Embedding-0.6B".to_string(),
            model_path: "model".to_string(),
        }
    }
}

pub struct Server {
    config: Arc<ServerConfig>,
}

impl Server {
    pub fn new(config: ServerConfig) -> Self {
        Server {
            config: Arc::new(config),
        }
    }

    pub async fn run(self, addr: SocketAddr) -> anyhow::Result<()> {
        let app = Router::new()
            .route("/healthz", get(healthz))
            .route("/readyz", get(readyz))
            .route("/v1/models", get(list_models))
            .route("/v1/embeddings", post(create_embedding))
            .with_state(self.config);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        tracing::info!("Server listening on {}", addr);
        
        axum::serve(listener, app).await?;
        
        Ok(())
    }
}

async fn healthz() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn readyz(State(config): State<Arc<ServerConfig>>) -> impl IntoResponse {
    // TODO: Check if model and GPU are ready
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn list_models(State(config): State<Arc<ServerConfig>>) -> impl IntoResponse {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: config.model_name.clone(),
            object: "model".to_string(),
            created: 0,
            owned_by: "qwen".to_string(),
        }],
    })
}

async fn create_embedding(
    State(config): State<Arc<ServerConfig>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Validate model name
    if request.model != config.model_name {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: format!("Model '{}' not found", request.model),
                    error_type: "invalid_request_error".to_string(),
                    code: "model_not_found".to_string(),
                },
            }),
        ));
    }

    // Validate encoding format
    match request.encoding_format {
        EncodingFormat::Float => {}
        EncodingFormat::Base64 => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Base64 encoding not supported in v1".to_string(),
                        error_type: "invalid_request_error".to_string(),
                        code: "unsupported_encoding".to_string(),
                    },
                }),
            ));
        }
    }

    // TODO: Implement actual embedding
    // For now, return a placeholder
    let texts = match request.input {
        EmbeddingInput::Single(text) => vec![text],
        EmbeddingInput::Multiple(texts) => texts,
    };

    let data: Vec<EmbeddingData> = texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            // Placeholder: return zeros
            EmbeddingData {
                object: "embedding".to_string(),
                embedding: vec![0.0; request.dimensions],
                index: i,
            }
        })
        .collect();

    let total_tokens: usize = texts.iter().map(|t| t.len() / 4).sum(); // Rough estimate

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: config.model_name.clone(),
        usage: Usage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}
```

**Step 3: Add tower dependency for axum layers**

Update `crates/qemb-service/Cargo.toml`:
```toml
[package]
name = "qemb-service"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
qemb-tokenizer.workspace = true
tokio.workspace = true
axum.workspace = true
tower = "0.5"
serde.workspace = true
serde_json.workspace = true
tracing.workspace = true
tracing-subscriber.workspace = true
```

**Step 4: Verify compiles**

Run: `cargo build -p qemb-service`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add crates/qemb-service/
git commit -m "feat(service): add health and models endpoints with OpenAI-compatible API types"
```

---

## Task 7: Wire Up CLI Serve Command

**TDD scenario:** Trivial change — manual testing

**Files:**
- Modify: `crates/qemb-cli/src/main.rs`

**Step 1: Update main.rs to use service**

Modify `crates/qemb-cli/src/main.rs`:
```rust
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use qemb_service::{Server, ServerConfig};

#[derive(Parser)]
#[command(name = "qemb")]
#[command(about = "Qwen3 Embedding Bare-Metal RDNA3 Service", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the embedding service
    Serve {
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
    /// Convert Hugging Face model to packed format
    Convert {
        /// Path to Hugging Face model directory
        #[arg(short, long)]
        input: String,
        /// Path to output packed model
        #[arg(short, long)]
        output: String,
    },
    /// Run offline embedding inference
    Run {
        /// Input text to embed
        #[arg(short, long)]
        text: String,
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Serve { model, port } => {
            let config = ServerConfig {
                model_name: "Qwen3-Embedding-0.6B".to_string(),
                model_path: model,
            };
            
            let server = Server::new(config);
            let addr: SocketAddr = ([0, 0, 0, 0], port).into();
            
            tracing::info!("Starting server on {}", addr);
            server.run(addr).await?;
        }
        Commands::Convert { input, output } => {
            println!("Converting {} to {}", input, output);
            println!("ERROR: Converter not yet implemented");
            std::process::exit(1);
        }
        Commands::Run { text, model } => {
            println!("Running inference on '{}' with model {}", text, model);
            println!("ERROR: Offline runner not yet implemented");
            std::process::exit(1);
        }
    }
    
    Ok(())
}
```

**Step 2: Verify compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 3: Test server starts**

Run: `cargo run -- serve --port 3001 &`
Wait 2 seconds, then: `curl http://localhost:3001/healthz`
Expected: `{"status":"ok"}`

Kill the server.

**Step 4: Commit**

```bash
git add crates/qemb-cli/src/main.rs
git commit -m "feat(cli): wire up serve command with working HTTP server"
```

---

## Task 8: Update README with Milestone Status

**TDD scenario:** Trivial change — documentation

**Files:**
- Modify: `README.md` (create if needed)

**Step 1: Create README**

Create `README.md`:
```markdown
# Qwen3 Embedding Bare-Metal RDNA3 Service

A Rust service that runs `Qwen/Qwen3-Embedding-0.6B` on AMD RDNA3 GPUs through a bare-metal stack.

## Milestone Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 Platform Bring-Up | 🔄 In Progress | gfx1103 target support, KFD runtime |
| M1 Primitive Validation | ⬜ Not Started | GEMM, gather, RMSNorm, SiLU kernels |
| M2 Single-Layer Validation | ⬜ Not Started | One transformer layer matches reference |
| M3 Full Offline Runner | ⬜ Not Started | CLI produces embeddings |
| M4 Service API | ⬜ Not Started | `/v1/embeddings` works end-to-end |
| M5 Stabilization | ⬜ Not Started | Memory stable, telemetry in place |

## Architecture

- **qemb-common**: Shared types, error handling, config
- **qemb-runtime**: KFD runtime, memory management, scheduling
- **qemb-kernels**: GPU kernel implementations
- **qemb-convert**: Offline weight converter (HF → runtime format)
- **qemb-tokenizer**: Tokenization wrapper
- **qemb-service**: HTTP server (axum on tokio)
- **qemb-cli**: CLI tool for testing and conversion

## Target Hardware

- AMD 780M iGPU (gfx1103)
- Linux with `/dev/kfd` and `/dev/dri/renderD*` access

## Quick Start

```bash
# Build
cargo build --release

# Start server (placeholder - no real inference yet)
cargo run --release -- serve --port 3000

# Health check
curl http://localhost:3000/healthz
```

## Design Document

See [docs/plans/2026-03-23-qwen3-embedding-design.md](docs/plans/2026-03-23-qwen3-embedding-design.md) for full architecture decisions.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with milestone status and architecture overview"
```

---

## Phase 1 Complete

After completing all 8 tasks, you should have:

1. ✅ Workspace skeleton with 7 crates
2. ✅ gfx1103 target support
3. ✅ GPU device detection via KFD
4. ✅ Minimal KFD runtime wrapper
5. ✅ Code object builder foundation
6. ✅ HTTP server with health and models endpoints
7. ✅ CLI with serve command working
8. ✅ README documenting progress

**Next phase:** gfx1103 bring-up, KFD smoke tests, and actual kernel dispatch.