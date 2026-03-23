//! AMD GPU code object (ELF) generation
//!
//! This module generates AMDGPU ELF code objects for GPU kernels.
//!
//! Based on refs/t0-gpu/src/rdna3_code_object.rs

use crate::target::GpuTarget;
use qemb_common::Result;

/// AMDGPU code object builder
pub struct CodeObjectBuilder {
    target: GpuTarget,
    kernels: Vec<KernelInfo>,
}

/// Information about a GPU kernel
#[derive(Debug, Clone)]
pub struct KernelInfo {
    /// Kernel name
    pub name: String,
    /// Compiled kernel code (ISA)
    pub code: Vec<u8>,
    /// Number of scalar registers
    pub sgpr_count: u16,
    /// Number of vector registers
    pub vgpr_count: u16,
    /// Shared memory (LDS) size in bytes
    pub shared_memory_bytes: u32,
}

impl CodeObjectBuilder {
    /// Create a new code object builder for the specified target
    pub fn new(target: GpuTarget) -> Self {
        CodeObjectBuilder {
            target,
            kernels: Vec::new(),
        }
    }

    /// Add a kernel to the code object
    pub fn add_kernel(&mut self, kernel: KernelInfo) -> &mut Self {
        self.kernels.push(kernel);
        self
    }

    /// Build the AMDGPU ELF code object
    pub fn build(&self) -> Result<Vec<u8>> {
        // Create ELF header for AMDGPU
        let mut elf = Vec::new();

        // ELF header (64 bytes)
        elf.extend_from_slice(&self.build_elf_header());

        // Program headers (placeholder for now)
        elf.extend_from_slice(&self.build_program_headers());

        // Section data (placeholder for now)
        elf.extend_from_slice(&self.build_sections());

        Ok(elf)
    }

    /// Get the e_flags for this target
    pub fn e_flags(&self) -> u32 {
        match self.target {
            // EF_AMDGPU_MACH values from ELF spec for AMDGPU
            // See: https://github.com/ROCm-Developer-Tools/ROCR-Runtime/blob/master/src/inc/amd_comgr.h
            GpuTarget::Gfx1100 => 0x1000, // EF_AMDGPU_MACH_AMDGCN_GFX1100
            GpuTarget::Gfx1103 => 0x1030, // EF_AMDGPU_MACH_AMDGCN_GFX1103
        }
    }

    /// Get the note vendor name
    pub fn note_vendor(&self) -> &'static [u8] {
        b"AMD"
    }

    /// Build the ELF header
    fn build_elf_header(&self) -> [u8; 64] {
        let mut header = [0u8; 64];

        // ELF magic number
        header[0..4].copy_from_slice(b"\x7fELF");

        // EI_CLASS: 64-bit ELF
        header[4] = 2;

        // EI_DATA: Little endian
        header[5] = 1;

        // EI_VERSION: ELF version 1
        header[6] = 1;

        // EI_OSABI: ELFOSABI_AMDGPU (0x40)
        header[7] = 0x40;

        // EI_ABIVERSION: 0
        header[8] = 0;

        // EI_PAD: padding (zeros, already set)

        // e_type: ET_REL (relocatable) = 1
        header[16..18].copy_from_slice(&(1u16).to_le_bytes());

        // e_machine: EM_AMDGPU (0xE0 = 224)
        header[18..20].copy_from_slice(&(0xE0u16).to_le_bytes());

        // e_version: 1
        header[20..24].copy_from_slice(&(1u32).to_le_bytes());

        // e_entry: 0 (no entry point for relocatable)
        header[24..32].copy_from_slice(&(0u64).to_le_bytes());

        // e_phoff: program header offset (right after ELF header = 64)
        header[32..40].copy_from_slice(&(0u64).to_le_bytes());

        // e_shoff: section header offset (placeholder)
        header[40..48].copy_from_slice(&(0u64).to_le_bytes());

        // e_flags: target-specific flags
        header[48..52].copy_from_slice(&self.e_flags().to_le_bytes());

        // e_ehsize: ELF header size = 64
        header[52..54].copy_from_slice(&(64u16).to_le_bytes());

        // e_phentsize: program header entry size = 56 (for 64-bit)
        header[54..56].copy_from_slice(&(56u16).to_le_bytes());

        // e_phnum: number of program headers (0 for now)
        header[56..58].copy_from_slice(&(0u16).to_le_bytes());

        // e_shentsize: section header entry size = 64 (for 64-bit)
        header[58..60].copy_from_slice(&(64u16).to_le_bytes());

        // e_shnum: number of section headers (0 for now)
        header[60..62].copy_from_slice(&(0u16).to_le_bytes());

        // e_shstrndx: section name string table index (0 for now)
        header[62..64].copy_from_slice(&(0u16).to_le_bytes());

        header
    }

    /// Build program headers (placeholder)
    fn build_program_headers(&self) -> Vec<u8> {
        // TODO: Implement proper program headers
        Vec::new()
    }

    /// Build sections (placeholder)
    fn build_sections(&self) -> Vec<u8> {
        // TODO: Implement proper sections
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_object_builder_e_flags_gfx1100() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1100);
        assert_eq!(builder.e_flags(), 0x1000);
    }

    #[test]
    fn test_code_object_builder_e_flags_gfx1103() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        assert_eq!(builder.e_flags(), 0x1030);
    }

    #[test]
    fn test_note_vendor() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        assert_eq!(builder.note_vendor(), b"AMD");
    }

    #[test]
    fn test_build_empty_code_object() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        let result = builder.build();
        assert!(result.is_ok());
        let elf = result.unwrap();
        // Check ELF magic
        assert_eq!(&elf[0..4], b"\x7fELF");
        // Check 64-bit
        assert_eq!(elf[4], 2);
        // Check little endian
        assert_eq!(elf[5], 1);
        // Check AMDGPU OSABI
        assert_eq!(elf[7], 0x40);
        // Check AMDGPU machine
        assert_eq!(u16::from_le_bytes([elf[18], elf[19]]), 0xE0);
    }

    #[test]
    fn test_elf_header_e_flags() {
        let builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        let elf = builder.build().unwrap();
        // e_flags is at offset 48
        let e_flags = u32::from_le_bytes([elf[48], elf[49], elf[50], elf[51]]);
        assert_eq!(e_flags, 0x1030);
    }

    #[test]
    fn test_add_kernel() {
        let mut builder = CodeObjectBuilder::new(GpuTarget::Gfx1103);
        let kernel = KernelInfo {
            name: "test_kernel".to_string(),
            code: vec![0, 1, 2, 3],
            sgpr_count: 16,
            vgpr_count: 32,
            shared_memory_bytes: 0,
        };
        builder.add_kernel(kernel);
        assert_eq!(builder.kernels.len(), 1);
        assert_eq!(builder.kernels[0].name, "test_kernel");
    }

    #[test]
    fn test_kernel_info() {
        let kernel = KernelInfo {
            name: "my_kernel".to_string(),
            code: vec![0xDE, 0xAD, 0xBE, 0xEF],
            sgpr_count: 20,
            vgpr_count: 64,
            shared_memory_bytes: 4096,
        };

        assert_eq!(kernel.name, "my_kernel");
        assert_eq!(kernel.code, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(kernel.sgpr_count, 20);
        assert_eq!(kernel.vgpr_count, 64);
        assert_eq!(kernel.shared_memory_bytes, 4096);
    }
}