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
        let tensor = self
            .bundle
            .tensors
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found: {}", name))?;

        let offset = tensor.offset;
        let expected_size = tensor.size_bytes;

        if data.len() != expected_size {
            anyhow::bail!(
                "Tensor {} size mismatch: expected {}, got {}",
                name,
                expected_size,
                data.len()
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

        let tensor = bundle
            .tensors
            .tensors
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