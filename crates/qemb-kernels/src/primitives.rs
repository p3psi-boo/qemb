use qemb_runtime::{DType, TensorDesc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Gather,
    Gemm,
    RmsNorm,
    Silu,
    ResidualAdd,
}

#[derive(Debug, Clone)]
pub struct PrimitiveSpec {
    pub kind: PrimitiveKind,
    pub inputs: Vec<TensorDesc>,
    pub output: TensorDesc,
}

#[derive(Debug, thiserror::Error)]
pub enum PrimitiveError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("dtype mismatch: {0}")]
    DType(String),
    #[error("invalid input: {0}")]
    Invalid(String),
}

pub type PrimitiveResult<T> = std::result::Result<T, PrimitiveError>;

impl PrimitiveSpec {
    pub fn new(kind: PrimitiveKind, inputs: Vec<TensorDesc>, output: TensorDesc) -> Self {
        Self { kind, inputs, output }
    }
}

pub fn expect_dtype(desc: &TensorDesc, expected: DType) -> PrimitiveResult<()> {
    if desc.dtype != expected {
        return Err(PrimitiveError::DType(format!(
            "expected {:?}, got {:?}",
            expected, desc.dtype
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expect_dtype_ok() {
        let desc = TensorDesc::new(vec![2, 2], DType::F32);
        assert!(expect_dtype(&desc, DType::F32).is_ok());
    }

    #[test]
    fn test_expect_dtype_err() {
        let desc = TensorDesc::new(vec![2, 2], DType::BF16);
        assert!(expect_dtype(&desc, DType::F32).is_err());
    }
}
