use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

/// Creates a seeded Xoshiro256** RNG for deterministic, reproducible simulations.
pub fn create_rng(seed: u64) -> Xoshiro256StarStar {
    Xoshiro256StarStar::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_deterministic_rng() {
        let mut rng1 = create_rng(42);
        let mut rng2 = create_rng(42);
        let values1: Vec<f64> = (0..100).map(|_| rng1.gen()).collect();
        let values2: Vec<f64> = (0..100).map(|_| rng2.gen()).collect();
        assert_eq!(values1, values2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut rng1 = create_rng(42);
        let mut rng2 = create_rng(43);
        let val1: f64 = rng1.gen();
        let val2: f64 = rng2.gen();
        assert_ne!(val1, val2);
    }
}
