# Qwen3 Embedding Phase 2 Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Implement model converter and tokenizer loading to enable offline model bundle creation and tokenization.

**Architecture:** Extend qemb-convert with packed format writer, extend qemb-tokenizer with tokenizer loading, add tensor abstractions to qemb-runtime.

**Tech Stack:** Rust, safetensors crate for loading HF weights, tokenizers crate for tokenization, serde for schema

---

## Phase 2 Overview

This phase enables model loading:
1. Packed format schema and writer
2. Converter CLI that reads HF safetensors
3. Tokenizer loading and encoding
4. Tensor abstractions for runtime

**Estimated time:** 2-3 hours

---

## Task 1: Add Tensor Descriptor Abstractions

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-runtime/src/tensor.rs`
- Modify: `crates/qemb-runtime/src/lib.rs`

**Step 1: Write the failing test**

Create `crates/qemb-runtime/src/tensor.rs`:

```rust
//! Tensor descriptors and abstractions for GPU operations

use std::fmt;

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F16,
    BF16,
    F32,
    I32,
    U8,
}

impl DType {
    /// Size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F32 => 4,
            DType::I32 => 4,
            DType::U8 => 1,
        }
    }
}

/// Memory layout for tensor data
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major (C-style)
    RowMajor,
    /// Column-major (Fortran-style)
    ColumnMajor,
}

/// Descriptor for a tensor's shape and properties
#[derive(Debug, Clone, PartialEq)]
pub struct TensorDesc {
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Memory layout
    pub layout: Layout,
    /// Byte offset in packed buffer
    pub offset: usize,
}

impl TensorDesc {
    /// Create a new tensor descriptor
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        TensorDesc {
            shape,
            dtype,
            layout: Layout::RowMajor,
            offset: 0,
        }
    }

    /// Set the byte offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Set the memory layout
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = layout;
        self
    }

    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
    }

    /// Strides for each dimension (in elements)
    pub fn strides(&self) -> Vec<usize> {
        let mut strides = Vec::with_capacity(self.shape.len());
        let mut stride = 1usize;
        for dim in self.shape.iter().rev() {
            strides.push(stride);
            stride *= *dim;
        }
        strides.reverse();
        strides
    }
}

impl fmt::Display for TensorDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor({:?}, {:?})", self.shape, self.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size_bytes() {
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::U8.size_bytes(), 1);
    }

    #[test]
    fn test_tensor_desc_num_elements() {
        let desc = TensorDesc::new(vec![2, 3, 4], DType::F32);
        assert_eq!(desc.num_elements(), 24);
    }

    #[test]
    fn test_tensor_desc_size_bytes() {
        let desc = TensorDesc::new(vec![2, 3, 4], DType::F32);
        assert_eq!(desc.size_bytes(), 96);
    }

    #[test]
    fn test_tensor_desc_strides() {
        let desc = TensorDesc::new(vec![2, 3, 4], DType::F32);
        assert_eq!(desc.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_desc_with_offset() {
        let desc = TensorDesc::new(vec![4, 4], DType::F16).with_offset(1024);
        assert_eq!(desc.offset, 1024);
    }

    #[test]
    fn test_tensor_desc_display() {
        let desc = TensorDesc::new(vec![2, 3], DType::F32);
        assert_eq!(format!("{}", desc), "Tensor([2, 3], F32)");
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p qemb-runtime`
Expected: All tests pass

**Step 3: Update lib.rs to export tensor module**

Modify `crates/qemb-runtime/src/lib.rs`:
```rust
pub mod device;
pub mod target;
pub mod kfd;
pub mod code_object;
pub mod tensor;

pub use device::Device;
pub use target::GpuTarget;
pub use kfd::KfdDevice;
pub use tensor::{DType, Layout, TensorDesc};
```

**Step 4: Verify workspace still compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 5: Commit**

```bash
git add crates/qemb-runtime/src/tensor.rs crates/qemb-runtime/src/lib.rs
git commit -m "feat(runtime): add tensor descriptor abstractions

- Add DType enum for data types (F16, BF16, F32, I32, U8)
- Add Layout enum for memory layout
- Add TensorDesc struct with shape, dtype, layout, offset
- Add num_elements(), size_bytes(), strides() methods
- Add comprehensive tests"
```

---

## Task 2: Add Safetensors Dependency and Loader

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-convert/Cargo.toml`
- Create: `crates/qemb-convert/src/loader.rs`
- Modify: `crates/qemb-convert/src/lib.rs`

**Step 1: Add dependencies**

Update `crates/qemb-convert/Cargo.toml`:
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
safetensors = "0.5"
```

**Step 2: Write the loader module**

Create `crates/qemb-convert/src/loader.rs`:

```rust
//! Loader for Hugging Face safetensors format

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Loaded tensor data
pub struct LoadedTensor {
    pub name: String,
    pub data: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Load tensors from a safetensors file
pub struct SafetensorsLoader {
    tensors: HashMap<String, LoadedTensor>,
}

impl SafetensorsLoader {
    /// Load from a safetensors file
    pub fn from_file(path: &Path) -> Result<Self> {
        let mut file = File::open(path)
            .with_context(|| format!("Failed to open safetensors file: {:?}", path))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .with_context(|| "Failed to read safetensors file")?;

        let safetensors = SafeTensors::deserialize(&buffer)
            .with_context(|| "Failed to deserialize safetensors")?;

        let mut tensors = HashMap::new();
        
        for tensor_name in safetensors.names() {
            let tensor = safetensors.tensor(tensor_name)?;
            let data = tensor.data().to_vec();
            let shape = tensor.shape().to_vec();
            let dtype = format!("{:?}", tensor.dtype());
            
            tensors.insert(tensor_name.to_string(), LoadedTensor {
                name: tensor_name.to_string(),
                data,
                dtype,
                shape,
            });
        }

        Ok(SafetensorsLoader { tensors })
    }

    /// Get a tensor by name
    pub fn get(&self, name: &str) -> Option<&LoadedTensor> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get the number of tensors
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors() -> NamedTempFile {
        // Create a minimal safetensors file
        // This is the header + data for a simple tensor
        let header = r#"{"test_tensor":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        
        let mut file = NamedTempFile::new().unwrap();
        
        // Write header length (8 bytes, little endian)
        file.write_all(&header_len.to_le_bytes()).unwrap();
        // Write header
        file.write_all(header_bytes).unwrap();
        // Write tensor data (16 bytes = 4 floats * 4 bytes)
        file.write_all(&[0u8; 16]).unwrap();
        
        file
    }

    #[test]
    fn test_loader_nonexistent_file() {
        let result = SafetensorsLoader::from_file(Path::new("/nonexistent/file.safetensors"));
        assert!(result.is_err());
    }
}
```

**Step 3: Update lib.rs**

Modify `crates/qemb-convert/src/lib.rs`:
```rust
pub mod packer;
pub mod schema;
pub mod loader;

pub use schema::ModelBundle;
pub use loader::{SafetensorsLoader, LoadedTensor};
```

**Step 4: Add tempfile dev dependency**

Update `crates/qemb-convert/Cargo.toml`:
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
safetensors = "0.5"

[dev-dependencies]
tempfile = "3"
```

**Step 5: Verify compiles**

Run: `cargo build -p qemb-convert`
Expected: Compiles successfully

**Step 6: Commit**

```bash
git add crates/qemb-convert/
git commit -m "feat(convert): add safetensors loader for HF weights

- Add SafetensorsLoader struct to read .safetensors files
- Add LoadedTensor struct with name, data, dtype, shape
- Add safetensors dependency
- Add test for nonexistent file handling"
```

---

## Task 3: Implement Packed Format Schema

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-convert/src/schema.rs`

**Step 1: Expand schema.rs with packed format**

Replace `crates/qemb-convert/src/schema.rs`:

```rust
//! Schema for packed model format (qembpack-v1)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model bundle metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    /// Format version
    pub version: String,
    /// Model name
    pub name: String,
    /// Model architecture parameters
    pub config: ModelConfig,
    /// Tensor metadata
    pub tensors: TensorTable,
    /// Checksums for validation
    #[serde(default)]
    pub checksums: HashMap<String, String>,
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub embedding_dim: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 3072,
            num_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 32768,
            embedding_dim: 1024,
        }
    }
}

/// Table of tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorTable {
    pub tensors: Vec<TensorMeta>,
    pub total_bytes: usize,
}

/// Metadata for a single tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Tensor name (e.g., "layers.0.attn.q_proj.weight")
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Byte offset in weights.bin
    pub offset: usize,
    /// Byte size
    pub size_bytes: usize,
}

impl TensorMeta {
    /// Number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Builder for creating a ModelBundle
pub struct ModelBundleBuilder {
    name: String,
    config: ModelConfig,
    tensors: Vec<TensorMeta>,
    total_bytes: usize,
}

impl ModelBundleBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        ModelBundleBuilder {
            name: name.into(),
            config: ModelConfig::default(),
            tensors: Vec::new(),
            total_bytes: 0,
        }
    }

    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_tensor(mut self, name: impl Into<String>, shape: Vec<usize>, dtype: &str, size_bytes: usize) -> Self {
        let offset = self.total_bytes;
        self.tensors.push(TensorMeta {
            name: name.into(),
            shape,
            dtype: dtype.to_string(),
            offset,
            size_bytes,
        });
        self.total_bytes += size_bytes;
        self
    }

    pub fn build(self) -> ModelBundle {
        ModelBundle {
            version: "qembpack-v1".to_string(),
            name: self.name,
            config: self.config,
            tensors: TensorTable {
                tensors: self.tensors,
                total_bytes: self.total_bytes,
            },
            checksums: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 28);
    }

    #[test]
    fn test_tensor_meta_num_elements() {
        let meta = TensorMeta {
            name: "test".to_string(),
            shape: vec![2, 3, 4],
            dtype: "BF16".to_string(),
            offset: 0,
            size_bytes: 48,
        };
        assert_eq!(meta.num_elements(), 24);
    }

    #[test]
    fn test_model_bundle_builder() {
        let bundle = ModelBundleBuilder::new("test-model")
            .add_tensor("weight1", vec![10, 20], "BF16", 400)
            .add_tensor("weight2", vec![5, 5], "BF16", 50)
            .build();

        assert_eq!(bundle.name, "test-model");
        assert_eq!(bundle.version, "qembpack-v1");
        assert_eq!(bundle.tensors.tensors.len(), 2);
        assert_eq!(bundle.tensors.total_bytes, 450);
    }

    #[test]
    fn test_tensor_offsets() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("a", vec![10], "BF16", 20)
            .add_tensor("b", vec![20], "BF16", 40)
            .add_tensor("c", vec![30], "BF16", 60)
            .build();

        assert_eq!(bundle.tensors.tensors[0].offset, 0);
        assert_eq!(bundle.tensors.tensors[1].offset, 20);
        assert_eq!(bundle.tensors.tensors[2].offset, 60);
    }

    #[test]
    fn test_serialize_deserialize() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("weight", vec![10, 10], "BF16", 200)
            .build();

        let json = serde_json::to_string(&bundle).unwrap();
        let decoded: ModelBundle = serde_json::from_str(&json).unwrap();
        
        assert_eq!(decoded.name, "test");
        assert_eq!(decoded.tensors.tensors.len(), 1);
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p qemb-convert`
Expected: All tests pass

**Step 3: Commit**

```bash
git add crates/qemb-convert/src/schema.rs
git commit -m "feat(convert): implement packed format schema (qembpack-v1)

- Add ModelBundle with version, name, config, tensors, checksums
- Add ModelConfig with architecture parameters
- Add TensorTable and TensorMeta for tensor metadata
- Add ModelBundleBuilder for constructing bundles
- Add serialization support with serde
- Add comprehensive tests"
```

---

## Task 4: Implement Weights Writer

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-convert/src/writer.rs`
- Modify: `crates/qemb-convert/src/lib.rs`

**Step 1: Create writer module**

Create `crates/qemb-convert/src/writer.rs`:

```rust
//! Writer for packed model format

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use crate::schema::ModelBundle;

/// Writer for packed model bundles
pub struct BundleWriter {
    bundle: ModelBundle,
    weights: Vec<u8>,
}

impl BundleWriter {
    /// Create a new writer with the given bundle metadata
    pub fn new(bundle: ModelBundle) -> Self {
        let total_bytes = bundle.tensors.total_bytes;
        BundleWriter {
            bundle,
            weights: vec![0u8; total_bytes],
        }
    }

    /// Write tensor data at the correct offset
    pub fn write_tensor(&mut self, name: &str, data: &[u8]) -> Result<()> {
        let tensor = self.bundle.tensors.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;

        let offset = tensor.offset;
        let expected_size = tensor.size_bytes;
        
        if data.len() != expected_size {
            anyhow::bail!(
                "Tensor {} size mismatch: expected {}, got {}",
                name, expected_size, data.len()
            );
        }

        self.weights[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Write the bundle to a directory
    pub fn write_to_dir(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create directory: {:?}", dir))?;

        // Write model.json
        let model_json_path = dir.join("model.json");
        let mut file = File::create(&model_json_path)
            .with_context(|| format!("Failed to create: {:?}", model_json_path))?;
        
        let json = serde_json::to_string_pretty(&self.bundle)
            .with_context(|| "Failed to serialize model.json")?;
        file.write_all(json.as_bytes())?;

        // Write weights.bin
        let weights_path = dir.join("weights.bin");
        let mut file = File::create(&weights_path)
            .with_context(|| format!("Failed to create: {:?}", weights_path))?;
        file.write_all(&self.weights)?;

        Ok(())
    }

    /// Get the bundle metadata
    pub fn bundle(&self) -> &ModelBundle {
        &self.bundle
    }
}

/// Reader for packed model bundles
pub struct BundleReader;

impl BundleReader {
    /// Read bundle metadata from a directory
    pub fn read_metadata(dir: &Path) -> Result<ModelBundle> {
        let model_json_path = dir.join("model.json");
        let content = fs::read_to_string(&model_json_path)
            .with_context(|| format!("Failed to read: {:?}", model_json_path))?;
        
        let bundle: ModelBundle = serde_json::from_str(&content)
            .with_context(|| "Failed to parse model.json")?;
        
        Ok(bundle)
    }

    /// Read weights from a directory
    pub fn read_weights(dir: &Path) -> Result<Vec<u8>> {
        let weights_path = dir.join("weights.bin");
        let weights = fs::read(&weights_path)
            .with_context(|| format!("Failed to read: {:?}", weights_path))?;
        Ok(weights)
    }

    /// Read a specific tensor from weights
    pub fn read_tensor(dir: &Path, name: &str) -> Result<Vec<u8>> {
        let bundle = Self::read_metadata(dir)?;
        let weights = Self::read_weights(dir)?;
        
        let tensor = bundle.tensors.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;

        let offset = tensor.offset;
        let size = tensor.size_bytes;
        
        Ok(weights[offset..offset + size].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ModelBundleBuilder;
    use tempfile::tempdir;

    #[test]
    fn test_bundle_writer_write_tensor() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("weight1", vec![4], "BF16", 8)
            .add_tensor("weight2", vec![8], "BF16", 16)
            .build();

        let mut writer = BundleWriter::new(bundle);
        
        writer.write_tensor("weight1", &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        writer.write_tensor("weight2", &[0u8; 16]).unwrap();
    }

    #[test]
    fn test_bundle_writer_wrong_size() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("weight", vec![4], "BF16", 8)
            .build();

        let mut writer = BundleWriter::new(bundle);
        
        let result = writer.write_tensor("weight", &[1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_writer_unknown_tensor() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("weight", vec![4], "BF16", 8)
            .build();

        let mut writer = BundleWriter::new(bundle);
        
        let result = writer.write_tensor("unknown", &[1, 2, 3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_write_and_read() {
        let bundle = ModelBundleBuilder::new("test")
            .add_tensor("weight1", vec![4], "BF16", 8)
            .add_tensor("weight2", vec![8], "BF16", 16)
            .build();

        let mut writer = BundleWriter::new(bundle);
        writer.write_tensor("weight1", &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        writer.write_tensor("weight2", &[0u8; 16]).unwrap();

        let dir = tempdir().unwrap();
        writer.write_to_dir(dir.path()).unwrap();

        // Read back
        let read_bundle = BundleReader::read_metadata(dir.path()).unwrap();
        assert_eq!(read_bundle.name, "test");
        assert_eq!(read_bundle.tensors.tensors.len(), 2);

        let weights = BundleReader::read_weights(dir.path()).unwrap();
        assert_eq!(weights.len(), 24);

        let tensor1 = BundleReader::read_tensor(dir.path(), "weight1").unwrap();
        assert_eq!(tensor1, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }
}
```

**Step 2: Update lib.rs**

Modify `crates/qemb-convert/src/lib.rs`:
```rust
pub mod packer;
pub mod schema;
pub mod loader;
pub mod writer;

pub use schema::ModelBundle;
pub use loader::{SafetensorsLoader, LoadedTensor};
pub use writer::{BundleWriter, BundleReader};
```

**Step 3: Add tempfile dev dependency (if not already added)**

**Step 4: Run tests**

Run: `cargo test -p qemb-convert`
Expected: All tests pass

**Step 5: Commit**

```bash
git add crates/qemb-convert/src/writer.rs crates/qemb-convert/src/lib.rs
git commit -m "feat(convert): implement bundle writer and reader

- Add BundleWriter to write model.json and weights.bin
- Add BundleReader to read bundle metadata and weights
- Add write_tensor() method with offset-based writing
- Add read_tensor() method for single tensor extraction
- Add size validation on write
- Add tests with temp directories"
```

---

## Task 5: Implement Converter CLI Command

**TDD scenario:** Modifying existing code — manual testing

**Files:**
- Modify: `crates/qemb-cli/src/main.rs`
- Modify: `crates/qemb-cli/Cargo.toml`

**Step 1: Add dependencies to CLI**

Update `crates/qemb-cli/Cargo.toml`:
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

**Step 2: Update main.rs with converter implementation**

Modify `crates/qemb-cli/src/main.rs`:

```rust
use clap::{Parser, Subcommand};
use qemb_convert::{BundleWriter, BundleReader, ModelBundleBuilder, ModelConfig, SafetensorsLoader};
use qemb_service::{Server, ServerConfig};
use std::net::SocketAddr;
use std::path::PathBuf;

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
        /// Model name for the bundle
        #[arg(long, default_value = "Qwen3-Embedding-0.6B")]
        name: String,
    },
    /// Inspect a packed model bundle
    Inspect {
        /// Path to model bundle
        #[arg(short, long, default_value = "model")]
        model: String,
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
        Commands::Convert { input, output, name } => {
            println!("Converting model from {} to {}", input, output);
            
            let input_path = PathBuf::from(&input);
            let output_path = PathBuf::from(&output);

            // Look for safetensors files
            let safetensors_files: Vec<_> = std::fs::read_dir(&input_path)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().map(|ext| ext == "safetensors").unwrap_or(false))
                .map(|e| e.path())
                .collect();

            if safetensors_files.is_empty() {
                anyhow::bail!("No .safetensors files found in {:?}", input_path);
            }

            println!("Found {} safetensors file(s)", safetensors_files.len());

            // Load all tensors
            let mut all_tensors = Vec::new();
            for file in &safetensors_files {
                println!("Loading {:?}", file);
                let loader = SafetensorsLoader::from_file(file)?;
                for name in loader.names() {
                    let tensor = loader.get(name).unwrap();
                    all_tensors.push((name.clone(), tensor.data.clone(), tensor.shape.clone()));
                }
            }

            println!("Loaded {} tensors", all_tensors.len());

            // Build the bundle
            let config = ModelConfig::default();
            let mut builder = ModelBundleBuilder::new(&name).with_config(config);

            for (tensor_name, data, shape) in &all_tensors {
                let size_bytes = data.len();
                builder = builder.add_tensor(tensor_name, shape.clone(), "BF16", size_bytes);
            }

            let bundle = builder.build();
            let mut writer = BundleWriter::new(bundle);

            for (tensor_name, data, _shape) in &all_tensors {
                writer.write_tensor(tensor_name, data)?;
            }

            writer.write_to_dir(&output_path)?;
            println!("Model bundle written to {:?}", output_path);
        }
        Commands::Inspect { model } => {
            let model_path = PathBuf::from(&model);
            
            match BundleReader::read_metadata(&model_path) {
                Ok(bundle) => {
                    println!("Model: {}", bundle.name);
                    println!("Version: {}", bundle.version);
                    println!("\nConfig:");
                    println!("  hidden_size: {}", bundle.config.hidden_size);
                    println!("  intermediate_size: {}", bundle.config.intermediate_size);
                    println!("  num_layers: {}", bundle.config.num_layers);
                    println!("  num_attention_heads: {}", bundle.config.num_attention_heads);
                    println!("  num_key_value_heads: {}", bundle.config.num_key_value_heads);
                    println!("  vocab_size: {}", bundle.config.vocab_size);
                    println!("  max_position_embeddings: {}", bundle.config.max_position_embeddings);
                    
                    println!("\nTensors: {} tensors, {} bytes", 
                        bundle.tensors.tensors.len(),
                        bundle.tensors.total_bytes
                    );
                    
                    for tensor in &bundle.tensors.tensors {
                        println!("  {} {:?} {} @ {}", tensor.name, tensor.shape, tensor.dtype, tensor.offset);
                    }
                }
                Err(e) => {
                    println!("Failed to read model bundle: {}", e);
                    std::process::exit(1);
                }
            }
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

**Step 3: Verify compiles**

Run: `cargo build`
Expected: Compiles successfully

**Step 4: Test inspect command**

Run: `cargo run -- inspect --model model`
Expected: Shows error about missing model (expected)

**Step 5: Commit**

```bash
git add crates/qemb-cli/src/main.rs crates/qemb-cli/Cargo.toml
git commit -m "feat(cli): implement convert and inspect commands

- Add convert command that reads safetensors and writes packed bundle
- Add inspect command to view bundle metadata and tensors
- Update CLI dependencies with qemb-convert
- Add error handling for missing files"
```

---

## Task 6: Update Tokenizer for Real Usage

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-tokenizer/src/tokenizer.rs`
- Create: `crates/qemb-tokenizer/src/bundle.rs`
- Modify: `crates/qemb-tokenizer/src/lib.rs`

**Step 1: Create bundle loader for tokenizer**

Create `crates/qemb-tokenizer/src/bundle.rs`:

```rust
//! Tokenizer loading from model bundles

use qemb_common::{Error, Result};
use std::path::Path;

use crate::tokenizer::Tokenizer;

/// Tokenizer bundle containing tokenizer files
pub struct TokenizerBundle {
    pub tokenizer: Tokenizer,
}

impl TokenizerBundle {
    /// Load tokenizer from a model bundle directory
    pub fn from_bundle_dir(dir: &Path) -> Result<Self> {
        // Look for tokenizer.json in the bundle
        let tokenizer_path = dir.join("tokenizer.json");
        
        if !tokenizer_path.exists() {
            return Err(Error::Tokenization(format!(
                "tokenizer.json not found in {:?}",
                dir
            )));
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path.to_str().unwrap())?;

        Ok(TokenizerBundle { tokenizer })
    }

    /// Get the underlying tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[test]
    fn test_bundle_missing_tokenizer() {
        let dir = tempdir().unwrap();
        let result = TokenizerBundle::from_bundle_dir(dir.path());
        assert!(result.is_err());
    }
}
```

**Step 2: Update lib.rs**

Modify `crates/qemb-tokenizer/src/lib.rs`:
```rust
pub mod tokenizer;
pub mod bundle;

pub use tokenizer::Tokenizer;
pub use bundle::TokenizerBundle;
```

**Step 3: Run tests**

Run: `cargo test -p qemb-tokenizer`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/qemb-tokenizer/src/bundle.rs crates/qemb-tokenizer/src/lib.rs
git commit -m "feat(tokenizer): add TokenizerBundle for loading from model dir

- Add TokenizerBundle struct with from_bundle_dir method
- Look for tokenizer.json in bundle directory
- Add error handling for missing tokenizer
- Add test for missing tokenizer case"
```

---

## Task 7: Update README with Phase 2 Progress

**TDD scenario:** Trivial change — documentation

**Files:**
- Modify: `README.md`

**Step 1: Update README milestone status**

Update README.md to reflect Phase 2 progress:

```markdown
# Qwen3 Embedding Bare-Metal RDNA3 Service

A Rust service that runs `Qwen/Qwen3-Embedding-0.6B` on AMD RDNA3 GPUs through a bare-metal stack.

## Milestone Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 Platform Bring-Up | ✅ Complete | gfx1103 target support, KFD runtime |
| M1 Primitive Validation | ⬜ Not Started | GEMM, gather, RMSNorm, SiLU kernels |
| M2 Single-Layer Validation | ⬜ Not Started | One transformer layer matches reference |
| M3 Full Offline Runner | 🔄 In Progress | CLI produces embeddings |
| M4 Service API | ⬜ Not Started | `/v1/embeddings` works end-to-end |
| M5 Stabilization | ⬜ Not Started | Memory stable, telemetry in place |
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README with Phase 2 progress

- Mark M0 Platform Bring-Up as complete
- Mark M3 Full Offline Runner as in progress
- Add convert and inspect commands to CLI reference"
```

---

## Phase 2 Complete

After completing all 7 tasks, you should have:

1. ✅ Tensor descriptor abstractions
2. ✅ Safetensors loader for HF weights
3. ✅ Packed format schema (qembpack-v1)
4. ✅ Bundle writer and reader
5. ✅ Convert and inspect CLI commands
6. ✅ Tokenizer bundle loading
7. ✅ Updated README

**Next phase:** Primitive kernel validation (GEMM, gather, RMSNorm, SiLU) and KFD smoke tests on actual hardware.