//! KFD (Kernel Fusion Driver) runtime for AMD GPUs
//!
//! This module provides direct access to AMD GPUs via /dev/kfd
//! without ROCm or HIP runtime dependencies.
//!
//! Based on refs/t0-gpu/src/kfd/mod.rs

use qemb_common::{Error, Result};
use std::fs::OpenOptions;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::Path;
use std::ptr;

// KFD ioctl numbers (from Linux kernel include/uapi/linux/kfd_ioctl.h)
const AMDKFD_IOC_GET_VERSION: u64 = 0x80084B01;
#[allow(dead_code)]
const AMDKFD_IOC_ACQUIRE_VM: u64 = 0x40084B15;
const AMDKFD_IOC_ALLOC_MEMORY: u64 = 0xC0284B16;
#[allow(dead_code)]
const AMDKFD_IOC_FREE_MEMORY: u64 = 0x40084B17;
#[allow(dead_code)]
const AMDKFD_IOC_MAP_MEMORY: u64 = 0xC0184B18;
#[allow(dead_code)]
const AMDKFD_IOC_CREATE_QUEUE: u64 = 0xC0604B02;
#[allow(dead_code)]
const AMDKFD_IOC_DESTROY_QUEUE: u64 = 0xC0084B03;
#[allow(dead_code)]
const AMDKFD_IOC_RUNTIME_ENABLE: u64 = 0xC0104B25;

// Memory allocation flags
pub const KFD_IOC_ALLOC_MEM_FLAGS_VRAM: u32 = 1 << 0;
pub const KFD_IOC_ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
pub const KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;
pub const KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
pub const KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC: u32 = 1 << 29;

// Queue types
pub const KFD_IOC_QUEUE_TYPE_COMPUTE: u32 = 0x0;
pub const KFD_IOC_QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;

/// KFD device handle
pub struct KfdDevice {
    fd: RawFd,
    gpu_id: u32,
}

impl KfdDevice {
    /// Open the KFD device at the specified path
    pub fn open(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)
            .map_err(|e| Error::Gpu(format!("Failed to open KFD device: {}", e)))?;

        let fd = file.as_raw_fd();
        // Prevent the file from being closed when it goes out of scope
        std::mem::forget(file);

        // Get GPU ID from sysfs
        let gpu_id = Self::detect_gpu_id()?;

        Ok(KfdDevice { fd, gpu_id })
    }

    /// Get the file descriptor
    pub fn fd(&self) -> RawFd {
        self.fd
    }

    /// Get the GPU ID
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Detect GPU ID from sysfs
    fn detect_gpu_id() -> Result<u32> {
        let topology_path = Path::new("/sys/class/kfd/kfd/topology/nodes");

        if !topology_path.exists() {
            return Err(Error::Gpu("KFD topology not found".to_string()));
        }

        for entry in std::fs::read_dir(topology_path)
            .map_err(|e| Error::Gpu(format!("Failed to read KFD topology: {}", e)))?
        {
            let entry = entry.map_err(|e| {
                Error::Gpu(format!("Failed to read topology entry: {}", e))
            })?;

            let node_path = entry.path();
            let name_path = node_path.join("name");

            if name_path.exists() {
                if let Ok(name) = std::fs::read_to_string(&name_path) {
                    // Skip CPU nodes
                    if name.trim() != "cpu" && !name.trim().is_empty() {
                        // The GPU ID is the node number
                        if let Some(node_str) = node_path.file_name() {
                            if let Ok(gpu_id) = node_str.to_string_lossy().parse::<u32>() {
                                return Ok(gpu_id);
                            }
                        }
                    }
                }
            }
        }

        Err(Error::Gpu("No GPU found in KFD topology".to_string()))
    }

    /// Get KFD version
    pub fn get_version(&self) -> Result<(u32, u32)> {
        #[repr(C)]
        struct KfdGetVersionArgs {
            major_version: u32,
            minor_version: u32,
        }

        let mut args = KfdGetVersionArgs {
            major_version: 0,
            minor_version: 0,
        };

        unsafe {
            let ret = libc::ioctl(
                self.fd,
                AMDKFD_IOC_GET_VERSION,
                &mut args as *mut _ as *mut libc::c_void,
            );
            if ret < 0 {
                return Err(Error::Gpu(format!(
                    "KFD get_version ioctl failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        Ok((args.major_version, args.minor_version))
    }

    /// Allocate GPU memory (VRAM or GTT)
    pub fn alloc_memory(&self, size: usize, flags: u32) -> Result<GpuBuffer> {
        #[repr(C)]
        struct KfdAllocMemoryArgs {
            va_addr: u64,
            size: u64,
            handle: u64,
            mmap_offset: u64,
            gpu_id: u32,
            flags: u32,
        }

        let mut args = KfdAllocMemoryArgs {
            va_addr: 0,
            size: size as u64,
            handle: 0,
            mmap_offset: 0,
            gpu_id: self.gpu_id,
            flags,
        };

        unsafe {
            let ret = libc::ioctl(
                self.fd,
                AMDKFD_IOC_ALLOC_MEMORY,
                &mut args as *mut _ as *mut libc::c_void,
            );
            if ret < 0 {
                return Err(Error::Gpu(format!(
                    "KFD alloc_memory ioctl failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
        }

        // Map the memory to host
        let host_ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd,
                args.mmap_offset as i64,
            )
        };

        if host_ptr == libc::MAP_FAILED {
            return Err(Error::Gpu("Failed to map GPU memory".to_string()));
        }

        Ok(GpuBuffer {
            va_addr: args.va_addr,
            host_ptr: host_ptr as *mut u8,
            size,
            handle: args.handle,
        })
    }

    /// Allocate VRAM memory
    pub fn alloc_vram(&self, size: usize) -> Result<GpuBuffer> {
        self.alloc_memory(
            size,
            KFD_IOC_ALLOC_MEM_FLAGS_VRAM
                | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC,
        )
    }

    /// Allocate GTT memory (host-visible, coherent)
    pub fn alloc_gtt(&self, size: usize) -> Result<GpuBuffer> {
        self.alloc_memory(
            size,
            KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE,
        )
    }
}

impl Drop for KfdDevice {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.fd);
        }
    }
}

/// GPU memory buffer
pub struct GpuBuffer {
    pub va_addr: u64,
    pub host_ptr: *mut u8,
    pub size: usize,
    pub handle: u64,
}

impl GpuBuffer {
    /// Write data to the buffer
    pub fn write(&self, data: &[u8]) {
        assert!(data.len() <= self.size);
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.host_ptr, data.len());
        }
    }

    /// Read data from the buffer
    pub fn read(&self, buf: &mut [u8]) {
        assert!(buf.len() <= self.size);
        unsafe {
            ptr::copy_nonoverlapping(self.host_ptr, buf.as_mut_ptr(), buf.len());
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.host_ptr.is_null() {
                libc::munmap(self.host_ptr as *mut libc::c_void, self.size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_kfd_device_open_nonexistent() {
        let result = KfdDevice::open(Path::new("/nonexistent/kfd"));
        assert!(result.is_err());
    }

    #[test]
    fn test_kfd_constants() {
        // Verify constants are defined correctly
        assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_VRAM, 1);
        assert_eq!(KFD_IOC_ALLOC_MEM_FLAGS_GTT, 2);
        assert!(KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE > 0);
    }

    #[test]
    fn test_kfd_device_probe() {
        // Try to open KFD device - this will fail in CI but should work on target hardware
        let result = KfdDevice::open(Path::new("/dev/kfd"));
        match result {
            Ok(device) => {
                // If we have a KFD device, test get_version
                println!("KFD device opened with GPU ID: {}", device.gpu_id());
                if let Ok((major, minor)) = device.get_version() {
                    println!("KFD version: {}.{}", major, minor);
                }
            }
            Err(e) => {
                // Expected in CI environments
                println!("KFD not available (expected in CI): {}", e);
            }
        }
    }
}