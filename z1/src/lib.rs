use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
pub mod multiplication_algorithms;
pub mod csv_io;
pub use crate::multiplication_algorithms::{binet_multiplication, strassen_multiplication,laderman_multiplication};



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