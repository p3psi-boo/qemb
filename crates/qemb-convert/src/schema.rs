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

    pub fn add_tensor(
        mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: &str,
        size_bytes: usize,
    ) -> Self {
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