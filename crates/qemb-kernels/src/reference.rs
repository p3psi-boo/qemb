use crate::primitives::{PrimitiveError, PrimitiveResult};

pub fn gemm_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
) -> PrimitiveResult<Vec<f32>> {
    if a.len() != m * k {
        return Err(PrimitiveError::Shape(format!(
            "A expected {} elems, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(PrimitiveError::Shape(format!(
            "B expected {} elems, got {}",
            k * n,
            b.len()
        )));
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
        return Err(PrimitiveError::Shape(format!(
            "residual add requires same len: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

pub fn gather_rows_f32(
    table: &[f32],
    rows: usize,
    cols: usize,
    indices: &[usize],
) -> PrimitiveResult<Vec<f32>> {
    if table.len() != rows * cols {
        return Err(PrimitiveError::Shape(format!(
            "table expected {} elems, got {}",
            rows * cols,
            table.len()
        )));
    }

    let mut out = Vec::with_capacity(indices.len() * cols);
    for &idx in indices {
        if idx >= rows {
            return Err(PrimitiveError::Invalid(format!(
                "gather index {} out of range {}",
                idx, rows
            )));
        }
        let start = idx * cols;
        out.extend_from_slice(&table[start..start + cols]);
    }
    Ok(out)
}

pub fn rmsnorm_f32(
    input: &[f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> PrimitiveResult<Vec<f32>> {
    if input.len() != rows * cols {
        return Err(PrimitiveError::Shape(format!(
            "input expected {} elems, got {}",
            rows * cols,
            input.len()
        )));
    }
    if weights.len() != cols {
        return Err(PrimitiveError::Shape(format!(
            "weights expected {} elems, got {}",
            cols,
            weights.len()
        )));
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
