// KFD runtime - implemented in Task 4

use qemb_common::{Error, Result};
use std::os::raw::c_int;
use std::path::Path;

pub struct KfdDevice {
    fd: c_int,
}

impl KfdDevice {
    pub fn open(_path: &Path) -> Result<Self> {
        Err(Error::Gpu("KFD not implemented".to_string()))
    }

    pub fn fd(&self) -> c_int {
        self.fd
    }
}