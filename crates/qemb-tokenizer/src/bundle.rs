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

    #[test]
    fn test_bundle_missing_tokenizer() {
        let dir = tempdir().unwrap();
        let result = TokenizerBundle::from_bundle_dir(dir.path());
        assert!(result.is_err());
    }
}