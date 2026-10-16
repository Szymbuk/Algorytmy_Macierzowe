use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
pub mod multiplication_algorithms;
use crate::multiplication_algorithms::{binet_multiplication, strassen_multiplication};


pub fn main1(){
    let m = generate_nxn_matrix(4);
    let m2 = generate_nxn_matrix(4);

    let res = binet_multiplication(&m, &m2);
    let res2 = strassen_multiplication(&m, &m2);

    println!("{:?}", res);
    print!("\n");
    println!("{:?}", res2);
}


pub fn generate_nxn_matrix(n: usize) -> Array<f64, ndarray::Dim<[usize; 2]>> {
    let matrix = Array::random((n,n), Uniform::try_from(0.00000001.. 1.0).unwrap());
    matrix
}

trait OperationCounter {
    fn increment_additions(&mut self, n: usize);
    fn increment_multiplications(&mut self, n: usize);
    fn increment_subtractions(&mut self, n: usize);
    fn increment_divisions(&mut self, n: usize);
    fn get_count(&self) -> (usize,usize,usize,usize);
}

struct EmptyCounter {}

impl OperationCounter for EmptyCounter {
    #[inline(always)]
    fn increment_additions(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_multiplications(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_subtractions(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_divisions(&mut self, _n: usize) {}

    #[inline(always)] 
    fn get_count(&self) -> (usize,usize,usize,usize) {
        (0,0,0,0)
    } 
}

struct Counter {
    count_additions: usize,
    count_multiplication: usize,
    count_subtraction: usize,
    count_division: usize,
}

impl OperationCounter for Counter {
    #[inline(always)]
    fn increment_additions(&mut self, n: usize) {
        self.count_additions += n;
    }

    #[inline(always)]
    fn increment_multiplications(&mut self, n: usize) {
        self.count_multiplication += n;
    }

    #[inline(always)]
    fn increment_subtractions(&mut self, n: usize) {
        self.count_subtraction += n;
    }

    #[inline(always)]
    fn increment_divisions(&mut self, n: usize) {
        self.count_division += n;
    }

    #[inline(always)]
    fn get_count(&self) -> (usize,usize,usize,usize) {
        (self.count_additions, self.count_multiplication, self.count_subtraction, self.count_division)
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