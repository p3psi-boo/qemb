use qemb_common::{Error, Result};
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| Error::Tokenization(format!("Failed to load tokenizer: {}", e)))?;
        Ok(Tokenizer { inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner
            .encode(text, false)
            .map_err(|e| Error::Tokenization(format!("Failed to encode: {}", e)))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn count_tokens(&self, text: &str) -> Result<usize> {
        let encoding = self.inner
            .encode(text, false)
            .map_err(|e| Error::Tokenization(format!("Failed to encode: {}", e)))?;
        Ok(encoding.len())
    }
}