use ndarray::Array;
use ndarray::s;


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


pub fn strassen_multiplication(a_array: &Array<f64, ndarray::Dim<[usize; 2]>>, b_array: &Array<f64, ndarray::Dim<[usize; 2]>>) -> Array<f64, ndarray::Dim<[usize; 2]>> {
    let a = a_array.view();
    let b = b_array.view();
    let n = a.nrows();
    let mut c = Array::zeros((n, n));
    
    if n <= 2 {
        return binet_multiplication(&a.to_owned(), &b.to_owned());
    }
    
    let mid = n / 2;
    
    let a00 = a.slice(s![..mid, ..mid]);
    let a01 = a.slice(s![..mid, mid..]);
    let a10 = a.slice(s![mid.., ..mid]);
    let a11 = a.slice(s![mid.., mid..]);
    
    let b00 = b.slice(s![..mid, ..mid]);
    let b01 = b.slice(s![..mid, mid..]);
    let b10 = b.slice(s![mid.., ..mid]);
    let b11 = b.slice(s![mid.., mid..]);
    
    let m0 = strassen_multiplication(&(a00.to_owned() + &a11.to_owned()), &(b00.to_owned() + &b11.to_owned()));
    let m1 = strassen_multiplication(&(a10.to_owned() + &a11.to_owned()), &b00.to_owned());
    let m2 = strassen_multiplication(&a00.to_owned(), &(b01.to_owned() - &b11.to_owned()));
    let m3 = strassen_multiplication(&a11.to_owned(), &(b10.to_owned() - &b00.to_owned()));
    let m4 = strassen_multiplication(&(a00.to_owned() + &a01.to_owned()), &b11.to_owned());
    let m5 = strassen_multiplication(&(a10.to_owned() - &a00.to_owned()), &(b00.to_owned() + &b01.to_owned()));
    let m6 = strassen_multiplication(&(a01.to_owned() - &a11.to_owned()), &(b10.to_owned() + &b11.to_owned()));


    c.slice_mut(s![..mid, ..mid]).assign(&(m0.clone() + m3.clone() + m6.clone() - m4.clone())); // C00
    c.slice_mut(s![mid.., ..mid]).assign(&(m1.clone() + m3.clone())); // C10
    c.slice_mut(s![..mid, mid..]).assign(&(m2.clone() + m4.clone())); // C01
    c.slice_mut(s![mid.., mid..]).assign(&(m0.clone() - m1.clone() + m2.clone() + m5.clone())); // C11

    c
}


#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_relative_eq;
    use ndarray::Zip;
    use crate::generate_nxn_matrix;
    use super::*;
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
        
    #[test]
    fn test_strassen_multiplication_power_of_two() {
        let n = 256;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = strassen_multiplication(&m1, &m2);



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }   


        #[test]
    fn test_strassen_multiplication_not_power_of_two() {
        let n = 256+1;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = strassen_multiplication(&m1, &m2);



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    } 
}