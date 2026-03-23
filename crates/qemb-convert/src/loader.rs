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

            tensors.insert(
                tensor_name.to_string(),
                LoadedTensor {
                    name: tensor_name.to_string(),
                    data,
                    dtype,
                    shape,
                },
            );
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

    #[test]
    fn test_loader_nonexistent_file() {
        let result = SafetensorsLoader::from_file(Path::new("/nonexistent/file.safetensors"));
        assert!(result.is_err());
    }
}