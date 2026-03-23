use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundle {
    pub name: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

impl Default for ModelBundle {
    fn default() -> Self {
        Self {
            name: "Qwen3-Embedding-0.6B".to_string(),
            hidden_size: 1024,
            intermediate_size: 3072,
            num_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            max_position_embeddings: 32768,
        }
    }
}