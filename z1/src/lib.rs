// use ndarray::Array;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray::{Array, Array2, s};

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

/// Algorytm Strassena z dynamicznym uzupełnianiem (paddingiem) do parzystego wymiaru:
/// 1. Warunek stopu: Jeśli wymiar macierzy n <= 3, algorytm przerywa rekurencję
///    i wykonuje tradycyjne, szybkie mnożenie iteracyjne za pomocą pętli.
/// 2. Sprawdzenie parzystości:
///    - Jeśli n jest parzyste, macierz dzielona jest bezpośrednio na 4 równe bloki.
///    - Jeśli n jest nieparzyste, macierz jest powiększana o 1 wiersz i 1 kolumnę
///      wypełnioną zerami (do rozmiaru n + 1), co pozwala na idealnie równy podział na bloki.
/// 3. Rekurencyjne obliczenie 7 iloczynów Strassena i złożenie macierzy wynikowej.
/// 4. Obcięcie wyniku: Jeśli oryginalny wymiar był nieparzysty, z końcowej macierzy
///    odrzucany jest dodany wcześniej zerowy wiersz i kolumna, zwracając dokładny wynik n x n.
pub fn strassen_multiplication(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();

    // 1. Warunek stopu dla małych macierzy
    if n <= 3 {
        let mut c = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    c[[i, j]] += a[[i, k]] * b[[k, j]];
                }
            }
        }
        return c;
    }

    // 2. Obsługa nieparzystego wymiaru (dynamiczne powiększenie do parzystego)
    let is_odd = n % 2 != 0;
    let actual_n = if is_odd { n + 1 } else { n };

    let a_work;
    let b_work;

    if is_odd {
        let mut a_padded = Array2::zeros((actual_n, actual_n));
        let mut b_padded = Array2::zeros((actual_n, actual_n));

        a_padded.slice_mut(s![0..n, 0..n]).assign(a);
        b_padded.slice_mut(s![0..n, 0..n]).assign(b);

        a_work = a_padded;
        b_work = b_padded;
    } else {
        a_work = a.to_owned();
        b_work = b.to_owned();
    }

    let mid = actual_n / 2;

    // Podział macierzy roboczych na 4 równe bloki
    let a11 = a_work.slice(s![0..mid, 0..mid]);
    let a12 = a_work.slice(s![0..mid, mid..actual_n]);
    let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
    let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);

    let b11 = b_work.slice(s![0..mid, 0..mid]);
    let b12 = b_work.slice(s![0..mid, mid..actual_n]);
    let b21 = b_work.slice(s![mid..actual_n, 0..mid]);
    let b22 = b_work.slice(s![mid..actual_n, mid..actual_n]);

    // 3. Obliczenie 7 iloczynów Strassena za pomocą wywołań rekurencyjnych
    // M1 = (A11 + A22) * (B11 + B22)
    let m1 = strassen_multiplication(&(a11.to_owned() + &a22), &(b11.to_owned() + &b22));
    // M2 = (A21 + A22) * B11
    let m2 = strassen_multiplication(&(a21.to_owned() + &a22), &b11.to_owned());
    // M3 = A11 * (B12 - B22)
    let m3 = strassen_multiplication(&a11.to_owned(), &(b12.to_owned() - &b22));
    // M4 = A22 * (B21 - B11)
    let m4 = strassen_multiplication(&a22.to_owned(), &(b21.to_owned() - &b11));
    // M5 = (A11 + A12) * B22
    let m5 = strassen_multiplication(&(a11.to_owned() + &a12), &b22.to_owned());
    // M6 = (A21 - A11) * (B11 + B12)
    let m6 = strassen_multiplication(&(a21.to_owned() - &a11), &(b11.to_owned() + &b12));
    // M7 = (A12 - A22) * (B21 + B22)
    let m7 = strassen_multiplication(&(a12.to_owned() - &a22), &(b21.to_owned() + &b22));

    // Składanie bloków wynikowych C
    let c11 = &m1 + &m4 - &m5 + &m7;
    let c12 = &m3 + &m5;
    let c21 = &m2 + &m4;
    let c22 = &m1 - &m2 + &m3 + &m6;

    let mut c_work = Array2::zeros((actual_n, actual_n));
    c_work.slice_mut(s![0..mid, 0..mid]).assign(&c11);
    c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&c12);
    c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&c21);
    c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&c22);

    // 4. Obcięcie wyniku do oryginalnego rozmiaru, jeśli wymiar wejściowy był nieparzysty
    if is_odd {
        c_work.slice(s![0..n, 0..n]).to_owned()
    } else {
        c_work
    }
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

    #[test]
    fn test_strassen_multiplication() {
        let n = 9;
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