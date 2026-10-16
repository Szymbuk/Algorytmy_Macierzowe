use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


pub fn main1(){
    let m = generate_nxn_matrix(3);
    let m2 = generate_nxn_matrix(3);
    let res = binet_multiplication(&m, &m2);
    println!("{:?}", res);
}


pub fn generate_nxn_matrix(n: usize) -> Array<f64, ndarray::Dim<[usize; 2]>> {
    let mut matrix = Array::random((n,n), Uniform::try_from(0.00000001.. 1.0).unwrap());
    matrix
}



pub fn binet_multiplication(a_array: &Array<f64, ndarray::Dim<[usize; 2]>>, b_array: &Array<f64, ndarray::Dim<[usize; 2]>>) -> Array<f64, ndarray::Dim<[usize; 2]>> {
    
    let a = a_array.view();
    let b = b_array.view();
    let mut c = Array::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    c
}




use assert_float_eq::assert_float_relative_eq;
use ndarray::Zip;

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

    
    #[test]
    fn test_binet_multiplication() {
        let n = 3;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = binet_multiplication(&m1, &m2);



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }   
}