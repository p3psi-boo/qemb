use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn random_f32(len: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| rng.random_range(-1.0f32..1.0f32))
        .collect()
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
