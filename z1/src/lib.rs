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