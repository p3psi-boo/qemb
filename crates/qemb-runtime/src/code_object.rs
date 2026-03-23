// Code object generation - implemented in Task 5

use crate::target::GpuTarget;
use qemb_common::{Error, Result};

pub struct CodeObjectBuilder {
    #[allow(dead_code)]
    target: GpuTarget,
}

impl CodeObjectBuilder {
    pub fn new(target: GpuTarget) -> Self {
        CodeObjectBuilder { target }
    }

    pub fn build(&self) -> Result<Vec<u8>> {
        Err(Error::Gpu("Code object generation not implemented".to_string()))
    }
}