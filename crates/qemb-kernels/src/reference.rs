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
