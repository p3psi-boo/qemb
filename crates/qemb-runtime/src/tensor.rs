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