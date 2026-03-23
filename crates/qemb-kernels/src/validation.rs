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
