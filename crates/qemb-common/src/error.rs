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