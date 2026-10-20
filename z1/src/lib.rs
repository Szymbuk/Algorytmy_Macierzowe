use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
pub mod multiplication_algorithms;
pub mod csv_io;
pub use crate::multiplication_algorithms::{binet_multiplication, strassen_multiplication,laderman_multiplication,dot_product};



pub fn generate_nxn_matrix(n: usize) -> Array2<f64> {
    let matrix = Array2::random((n,n), Uniform::try_from(0.00000001.. 1.0).unwrap());
    matrix
}


pub fn generate_seeded_matrix(n: usize, rng: &mut ChaCha8Rng) -> ndarray::Array2<f64> {
    let mut matrix = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] = rng.random_range(0.00000001..1.0);
        }
    }
    matrix
}

pub fn generate_sizes(max_n: usize) -> Vec<usize> {
    let mut sizes = Vec::new();

    // 1. Baza: rzadszy krok liniowy dla ogólnego trendu (np. co 10 lub 20)
    for i in (10..=max_n).step_by(10) {
        sizes.push(i);
    }

    // 2. Wstrzyknięcie potęg 2 i punktów "zaraz obok" (dla Strassena)
    let powers_2 = [16, 32, 64, 128, 256, 512, 1024];
    for &p in &powers_2 {
        if p <= max_n {
            sizes.push(p - 1); // Zaraz przed (najgorszy przypadek paddingu)
            sizes.push(p);     // Idealna potęga (zero paddingu)
            sizes.push(p + 1); // Zaraz po (znowu duży padding)
        }
    }

    // 3. Wstrzyknięcie potęg 3 i punktów "zaraz obok" (dla Ladermana)
    let powers_3 = [27, 81, 243, 729];
    for &p in &powers_3 {
        if p <= max_n {
            sizes.push(p - 1);
            sizes.push(p);
            sizes.push(p + 1);
        }
    }

    // 4. Sortowanie i usunięcie duplikatów
    sizes.sort_unstable();
    sizes.dedup();

    sizes
}



pub trait OperationCounter {
    fn new() -> Self;
    fn increment_additions(&mut self, n: usize);
    fn increment_multiplications(&mut self, n: usize);
    fn increment_divisions(&mut self, n: usize);
    fn get_count(&self) -> (usize,usize,usize);
}

pub struct EmptyCounter {}

impl OperationCounter for EmptyCounter {
    fn new() -> Self {
        EmptyCounter {}
    }

    #[inline(always)]
    fn increment_additions(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_multiplications(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_divisions(&mut self, _n: usize) {}

    #[inline(always)] 
    fn get_count(&self) -> (usize,usize,usize) {
        (0,0,0)
    } 
}

pub struct Counter {
    count_additions: usize,
    count_multiplication: usize,
    count_division: usize,
}

impl OperationCounter for Counter {
    fn new() -> Self {
        Counter { count_additions: 0, count_multiplication: 0, count_division: 0 }
    }


    #[inline(always)]
    fn increment_additions(&mut self, n: usize) {
        self.count_additions += n;
    }

    #[inline(always)]
    fn increment_multiplications(&mut self, n: usize) {
        self.count_multiplication += n;
    }

    #[inline(always)]
    fn increment_divisions(&mut self, n: usize) {
        self.count_division += n;
    }

    #[inline(always)]
    fn get_count(&self) -> (usize,usize,usize) {
        (self.count_additions, self.count_multiplication, self.count_division)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix_generation() {
        let n = 10;
        let m1 = generate_nxn_matrix(n);
        assert!(m1.shape() == &[n, n]);
        assert!(m1.iter().all(|x| *x >= 0.00000001 && *x < 1.0));
    }   
}