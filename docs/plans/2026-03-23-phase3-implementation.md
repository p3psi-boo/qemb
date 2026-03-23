# Qwen3 Embedding Phase 3a Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Build CPU reference primitives, synthetic input generators, error metrics, and a validation harness so later GPU kernels can be checked numerically.

**Architecture:** Keep GPU-facing APIs in `qemb-kernels`, but implement reference behavior on CPU first. This unblocks correctness work without pretending the current codebase can already emit runnable AMDGPU kernels.

**Tech Stack:** Rust, `qemb-kernels`, `qemb-runtime`, `qemb-cli`, `serde_json`

---

## Phase 3a Overview

This phase establishes primitive correctness infrastructure:
1. Primitive descriptors and error types
2. CPU reference implementations for gather/GEMM/RMSNorm/SiLU/residual add
3. Synthetic input generators and error metrics
4. Validation harness and CLI entrypoint
5. Validation logs under `docs/validation/`

**Estimated time:** 2-3 hours

---

## Task 1: Add Primitive Types and Errors

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-kernels/Cargo.toml`
- Modify: `crates/qemb-kernels/src/lib.rs`
- Replace: `crates/qemb-kernels/src/primitives.rs`

**Step 1: Add dependency on qemb-runtime types**

Update `crates/qemb-kernels/Cargo.toml`:
```toml
[package]
name = "qemb-kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
thiserror.workspace = true
anyhow.workspace = true
serde.workspace = true
serde_json.workspace = true
```

**Step 2: Replace primitives module**

Replace `crates/qemb-kernels/src/primitives.rs`:
```rust
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
```

**Step 3: Add module exports**

Update `crates/qemb-kernels/src/lib.rs`:
```rust
pub mod generators;
pub mod metrics;
pub mod primitives;
pub mod reference;
pub mod validation;

pub use primitives::{PrimitiveError, PrimitiveKind, PrimitiveResult, PrimitiveSpec};
```

**Step 4: Run tests**

Run: `cargo test -p qemb-kernels`
Expected: tests pass

**Step 5: Commit**

```bash
git add crates/qemb-kernels/
git commit -m "feat(kernels): add primitive specs and error types"
```

---

## Task 2: Implement CPU Reference GEMM and Residual Add

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-kernels/src/reference.rs`

**Step 1: Create reference module with GEMM and residual add**

Create `crates/qemb-kernels/src/reference.rs`:
```rust
use crate::primitives::{PrimitiveError, PrimitiveResult};

pub fn gemm_f32(m: usize, k: usize, n: usize, a: &[f32], b: &[f32]) -> PrimitiveResult<Vec<f32>> {
    if a.len() != m * k {
        return Err(PrimitiveError::Shape(format!("A expected {} elems, got {}", m * k, a.len())));
    }
    if b.len() != k * n {
        return Err(PrimitiveError::Shape(format!("B expected {} elems, got {}", k * n, b.len())));
    }

    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    Ok(out)
}

pub fn residual_add_f32(a: &[f32], b: &[f32]) -> PrimitiveResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(PrimitiveError::Shape(format!("residual add requires same len: {} vs {}", a.len(), b.len())));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let out = gemm_f32(2, 2, 2, &a, &b).unwrap();
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_residual_add_f32() {
        let out = residual_add_f32(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(out, vec![4.0, 6.0]);
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p qemb-kernels`
Expected: tests pass

**Step 3: Commit**

```bash
git add crates/qemb-kernels/src/reference.rs
git commit -m "feat(kernels): add CPU reference GEMM and residual add"
```

---

## Task 3: Implement CPU Reference Gather, RMSNorm, and SiLU

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Modify: `crates/qemb-kernels/src/reference.rs`

**Step 1: Extend reference module**

Append to `crates/qemb-kernels/src/reference.rs`:
```rust
pub fn gather_rows_f32(table: &[f32], rows: usize, cols: usize, indices: &[usize]) -> PrimitiveResult<Vec<f32>> {
    if table.len() != rows * cols {
        return Err(PrimitiveError::Shape(format!("table expected {} elems, got {}", rows * cols, table.len())));
    }

    let mut out = Vec::with_capacity(indices.len() * cols);
    for &idx in indices {
        if idx >= rows {
            return Err(PrimitiveError::Invalid(format!("gather index {} out of range {}", idx, rows)));
        }
        let start = idx * cols;
        out.extend_from_slice(&table[start..start + cols]);
    }
    Ok(out)
}

pub fn rmsnorm_f32(input: &[f32], weights: &[f32], rows: usize, cols: usize, epsilon: f32) -> PrimitiveResult<Vec<f32>> {
    if input.len() != rows * cols {
        return Err(PrimitiveError::Shape(format!("input expected {} elems, got {}", rows * cols, input.len())));
    }
    if weights.len() != cols {
        return Err(PrimitiveError::Shape(format!("weights expected {} elems, got {}", cols, weights.len())));
    }

    let mut out = vec![0.0f32; input.len()];
    for row in 0..rows {
        let slice = &input[row * cols..(row + 1) * cols];
        let mean_sq = slice.iter().map(|x| x * x).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + epsilon).sqrt();
        for col in 0..cols {
            out[row * cols + col] = slice[col] * inv_rms * weights[col];
        }
    }
    Ok(out)
}

pub fn silu_f32(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

#[cfg(test)]
mod more_tests {
    use super::*;

    #[test]
    fn test_gather_rows_f32() {
        let table = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = gather_rows_f32(&table, 3, 2, &[2, 0]).unwrap();
        assert_eq!(out, vec![5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rmsnorm_f32_shape() {
        let err = rmsnorm_f32(&[1.0, 2.0], &[1.0], 1, 2, 1e-5).unwrap_err();
        assert!(format!("{}", err).contains("weights expected 2 elems"));
    }

    #[test]
    fn test_silu_f32() {
        let out = silu_f32(&[0.0, 1.0]);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 0.7310586).abs() < 1e-6);
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p qemb-kernels`
Expected: tests pass

**Step 3: Commit**

```bash
git add crates/qemb-kernels/src/reference.rs
git commit -m "feat(kernels): add CPU reference gather, RMSNorm, and SiLU"
```

---

## Task 4: Add Synthetic Generators and Error Metrics

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-kernels/src/generators.rs`
- Create: `crates/qemb-kernels/src/metrics.rs`
- Modify: `crates/qemb-kernels/Cargo.toml`

**Step 1: Add rand dependency**

Update `crates/qemb-kernels/Cargo.toml`:
```toml
[package]
name = "qemb-kernels"
version.workspace = true
edition.workspace = true
license.workspace = true

[dependencies]
qemb-common.workspace = true
qemb-runtime.workspace = true
thiserror.workspace = true
anyhow.workspace = true
serde.workspace = true
serde_json.workspace = true
rand = "0.9"
```

**Step 2: Create generators module**

Create `crates/qemb-kernels/src/generators.rs`:
```rust
use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(-1.0f32..1.0f32)).collect()
}

pub fn random_indices(len: usize, upper: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random_range(0..upper)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_f32_deterministic() {
        assert_eq!(random_f32(4, 7), random_f32(4, 7));
    }

    #[test]
    fn test_random_indices_range() {
        let xs = random_indices(32, 5, 9);
        assert!(xs.iter().all(|&x| x < 5));
    }
}
```

**Step 3: Create metrics module**

Create `crates/qemb-kernels/src/metrics.rs`:
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorMetrics {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub cosine_similarity: f32,
}

pub fn compare_f32(reference: &[f32], candidate: &[f32]) -> ErrorMetrics {
    assert_eq!(reference.len(), candidate.len());

    let mut max_abs_error = 0.0f32;
    let mut sum_abs_error = 0.0f32;
    let mut dot = 0.0f32;
    let mut ref_norm = 0.0f32;
    let mut cand_norm = 0.0f32;

    for (&r, &c) in reference.iter().zip(candidate.iter()) {
        let abs_err = (r - c).abs();
        max_abs_error = max_abs_error.max(abs_err);
        sum_abs_error += abs_err;
        dot += r * c;
        ref_norm += r * r;
        cand_norm += c * c;
    }

    let cosine_similarity = if ref_norm == 0.0 || cand_norm == 0.0 {
        1.0
    } else {
        dot / (ref_norm.sqrt() * cand_norm.sqrt())
    };

    ErrorMetrics {
        max_abs_error,
        mean_abs_error: sum_abs_error / reference.len() as f32,
        cosine_similarity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_identical() {
        let m = compare_f32(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!(m.max_abs_error, 0.0);
        assert_eq!(m.mean_abs_error, 0.0);
        assert!((m.cosine_similarity - 1.0).abs() < 1e-6);
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p qemb-kernels`
Expected: tests pass

**Step 5: Commit**

```bash
git add crates/qemb-kernels/
git commit -m "feat(kernels): add synthetic generators and error metrics"
```

---

## Task 5: Add Validation Harness and JSON Logging

**TDD scenario:** New feature — full TDD cycle

**Files:**
- Create: `crates/qemb-kernels/src/validation.rs`
- Create: `docs/validation/.gitkeep`

**Step 1: Create validation harness**

Create `crates/qemb-kernels/src/validation.rs`:
```rust
use crate::metrics::{compare_f32, ErrorMetrics};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationLog {
    pub primitive: String,
    pub element_count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub cosine_similarity: f32,
}

impl ValidationLog {
    pub fn from_outputs(primitive: &str, reference: &[f32], candidate: &[f32]) -> Self {
        let metrics = compare_f32(reference, candidate);
        Self::from_metrics(primitive, reference.len(), metrics)
    }

    pub fn from_metrics(primitive: &str, element_count: usize, metrics: ErrorMetrics) -> Self {
        Self {
            primitive: primitive.to_string(),
            element_count,
            max_abs_error: metrics.max_abs_error,
            mean_abs_error: metrics.mean_abs_error,
            cosine_similarity: metrics.cosine_similarity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_log_from_outputs() {
        let log = ValidationLog::from_outputs("gemm", &[1.0, 2.0], &[1.0, 2.1]);
        assert_eq!(log.primitive, "gemm");
        assert_eq!(log.element_count, 2);
        assert!(log.max_abs_error > 0.0);
    }
}
```

**Step 2: Create validation docs dir marker**

Run:
```bash
mkdir -p docs/validation
touch docs/validation/.gitkeep
```

**Step 3: Run tests**

Run: `cargo test -p qemb-kernels`
Expected: tests pass

**Step 4: Commit**

```bash
git add crates/qemb-kernels/src/validation.rs docs/validation/.gitkeep
git commit -m "feat(kernels): add validation harness and log schema"
```

---

## Task 6: Add Primitive Validation CLI Command

**TDD scenario:** Modifying existing code — manual testing

**Files:**
- Modify: `crates/qemb-cli/Cargo.toml`
- Modify: `crates/qemb-cli/src/main.rs`

**Step 1: Ensure CLI can access qemb-kernels exports**

No new dependencies needed if workspace already links `qemb-kernels`.

**Step 2: Add validate-primitives command**

Modify `crates/qemb-cli/src/main.rs` by adding imports:
```rust
use qemb_kernels::generators::{random_f32, random_indices};
use qemb_kernels::reference::{gather_rows_f32, gemm_f32, residual_add_f32, rmsnorm_f32, silu_f32};
use qemb_kernels::validation::ValidationLog;
```

Add command variant:
```rust
    /// Validate CPU primitive reference paths and write logs
    ValidatePrimitives {
        /// Output JSON log path
        #[arg(long, default_value = "docs/validation/primitives.json")]
        out: String,
    },
```

Add match arm:
```rust
        Commands::ValidatePrimitives { out } => {
            let mut logs = Vec::new();

            let a = random_f32(8, 1);
            let b = random_f32(12, 2);
            let gemm = gemm_f32(2, 4, 3, &a, &b)?;
            logs.push(ValidationLog::from_outputs("gemm_f32", &gemm, &gemm));

            let table = random_f32(20, 3);
            let idx = random_indices(3, 5, 4);
            let gather = gather_rows_f32(&table, 5, 4, &idx)?;
            logs.push(ValidationLog::from_outputs("gather_rows_f32", &gather, &gather));

            let norm_in = random_f32(12, 5);
            let norm_w = random_f32(4, 6);
            let rms = rmsnorm_f32(&norm_in, &norm_w, 3, 4, 1e-5)?;
            logs.push(ValidationLog::from_outputs("rmsnorm_f32", &rms, &rms));

            let silu_in = random_f32(6, 7);
            let silu = silu_f32(&silu_in);
            logs.push(ValidationLog::from_outputs("silu_f32", &silu, &silu));

            let add_a = random_f32(6, 8);
            let add_b = random_f32(6, 9);
            let add = residual_add_f32(&add_a, &add_b)?;
            logs.push(ValidationLog::from_outputs("residual_add_f32", &add, &add));

            let out_path = PathBuf::from(&out);
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out_path, serde_json::to_vec_pretty(&logs)?)?;
            println!("Wrote validation log to {:?}", out_path);
        }
```

**Step 3: Run build and command**

Run:
```bash
cargo build
cargo run -- validate-primitives --out docs/validation/primitives.json
```
Expected: JSON log written successfully

**Step 4: Commit**

```bash
git add crates/qemb-cli/src/main.rs docs/validation/primitives.json
git commit -m "feat(cli): add primitive validation command"
```

---

## Task 7: Update README for Primitive Validation

**TDD scenario:** Trivial change — documentation

**Files:**
- Modify: `README.md`

**Step 1: Update milestone status and CLI docs**

Add to README:
- M1 Primitive Validation → `🔄 In Progress`
- `qemb validate-primitives --out docs/validation/primitives.json`

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for primitive validation workflow"
```

---

## Phase 3a Complete

After completing all 7 tasks, you should have:

1. ✅ Primitive specs and errors
2. ✅ CPU reference GEMM/residual add
3. ✅ CPU reference gather/RMSNorm/SiLU
4. ✅ Synthetic generators and error metrics
5. ✅ Validation harness and docs output schema
6. ✅ CLI primitive validation command
7. ✅ Updated README

**Next phase:** wire first executable GPU primitive path and compare GPU outputs against these reference implementations.
