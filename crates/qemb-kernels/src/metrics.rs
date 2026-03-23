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
