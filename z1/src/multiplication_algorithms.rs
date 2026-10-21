use ndarray::Array2;
use ndarray::s;

use crate::{OperationCounter};

pub fn binet_multiplication(a_array: &Array2<f64>, b_array: &Array2<f64>,counter: &mut impl OperationCounter) -> Array2<f64> {
    let a = a_array.view();
    let b = b_array.view();
    let mut c = Array2::zeros((a.nrows(), b.ncols()));
    counter.increment_multiplications(a.nrows() * b.ncols() * a.ncols());
    counter.increment_additions(a.nrows() * b.ncols() * (a.ncols() - 1));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            for k in 0..a.ncols() {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    c
}

pub fn binet_recursive_multiplication(a : &Array2<f64>, b: &Array2<f64>,counter: &mut impl OperationCounter) -> Array2<f64> {
    let n = a.nrows();

    if n <= 2 {
        return binet_multiplication(a, b, counter);
    }

    let (a_work, is_odd, original_n) = make_even(a);
    let (b_work, _, _) = make_even(b);

    let actual_n = a_work.nrows();

    let mid = actual_n / 2;

    let a11 = a_work.slice(s![0..mid, 0..mid]);
    let a12 = a_work.slice(s![0..mid, mid..actual_n]);
    let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
    let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);

    let b11 = b_work.slice(s![0..mid, 0..mid]);
    let b12 = b_work.slice(s![0..mid, mid..actual_n]);
    let b21 = b_work.slice(s![mid..actual_n, 0..mid]);
    let b22 = b_work.slice(s![mid..actual_n, mid..actual_n]);

    let term1 = binet_recursive_multiplication(&a11.to_owned(), &b11.to_owned(),counter);
    let term2 = binet_recursive_multiplication(&a12.to_owned(), &b21.to_owned(),counter);
    let c11 = &term1 + &term2;

    let term3 = binet_recursive_multiplication(&a11.to_owned(), &b12.to_owned(),counter);
    let term4 = binet_recursive_multiplication(&a12.to_owned(), &b22.to_owned(),counter);
    let c12 = &term3 + &term4;

    let term5 = binet_recursive_multiplication(&a21.to_owned(), &b11.to_owned(),counter);
    let term6 = binet_recursive_multiplication(&a22.to_owned(), &b21.to_owned(),counter);
    let c21 = &term5 + &term6;

    let term7 = binet_recursive_multiplication(&a21.to_owned(), &b12.to_owned(),counter);
    let term8 = binet_recursive_multiplication(&a22.to_owned(), &b22.to_owned(),counter);
    let c22 = &term7 + &term8;

    let mut c_work = Array2::zeros((actual_n, actual_n));
    c_work.slice_mut(s![0..mid, 0..mid]).assign(&c11);
    c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&c12);
    c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&c21);
    c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&c22);

    if is_odd {
        c_work.slice(s![0..n, 0..n]).to_owned()
    } else {
        c_work
    }
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
pub fn strassen_multiplication(a: &Array2<f64>, b: &Array2<f64>, counter: &mut impl OperationCounter) -> Array2<f64> {
    let n = a.nrows();

    // 1. Warunek stopu dla małych macierzy
    if n <= 16 {
        return binet_multiplication(a, b, counter);
    }

    // 2. Obsługa nieparzystego wymiaru (dynamiczne powiększenie do parzystego)
    let (a_work, is_odd, original_n) = make_even(a);
    let (b_work, _, _) = make_even(b);

    let actual_n = a_work.nrows();

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
    let m1 = strassen_multiplication(&(a11.to_owned() + &a22), &(b11.to_owned() + &b22),counter);
    // M2 = (A21 + A22) * B11
    let m2 = strassen_multiplication(&(a21.to_owned() + &a22), &b11.to_owned(),counter);
    // M3 = A11 * (B12 - B22)
    let m3 = strassen_multiplication(&a11.to_owned(), &(b12.to_owned() - &b22),counter);
    // M4 = A22 * (B21 - B11)
    let m4 = strassen_multiplication(&a22.to_owned(), &(b21.to_owned() - &b11),counter);
    // M5 = (A11 + A12) * B22
    let m5 = strassen_multiplication(&(a11.to_owned() + &a12), &b22.to_owned(),counter);
    // M6 = (A21 - A11) * (B11 + B12)
    let m6 = strassen_multiplication(&(a21.to_owned() - &a11), &(b11.to_owned() + &b12),counter);
    // M7 = (A12 - A22) * (B21 + B22)
    let m7 = strassen_multiplication(&(a12.to_owned() - &a22), &(b21.to_owned() + &b22),counter);

    // Składanie bloków wynikowych C
    let c11 = &m1 + &m4 - &m5 + &m7;
    let c12 = &m3 + &m5;
    let c21 = &m2 + &m4;
    let c22 = &m1 - &m2 + &m3 + &m6;

    counter.increment_additions(a11.nrows() * a11.ncols() * 18);

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
// funkcja pomocznicza do sprawdzenia i tworzenia parzystych macierzy
pub fn make_even(a: &Array2<f64>) -> (Array2<f64>, bool, usize) {
    let n = a.nrows();
    let is_odd = n % 2 != 0;

    if is_odd {
        let actual_n = n + 1;
        let mut a_padded = Array2::zeros((actual_n, actual_n));
        a_padded.slice_mut(s![0..n, 0..n]).assign(a);
        (a_padded, true, n)
    } else {
        (a.to_owned(), false, n)
    }
}
/// Rekurencyjny algorytm Ladermana dla macierzy dowolnego rozmiaru:
/// 1. Warunek stopu: Jeśli n <= 3, stosujemy bezpośrednie obliczenia (wzory Ladermana dla 3x3 lub pętle).
/// 2. Dopełnienie (Padding): Ponieważ macierz dzielimy na 3 części, rozmiar macierzy
///    powinien być powiększony do najbliższej wielokrotności/potęgi liczby 3.
/// 3. Podział na 9 bloków (3x3 podmacierze) i wykonanie 23 rekurencyjnych mnożeń.
/// 4. Złożenie bloków wynikowych C oraz obcięcie do oryginalnego rozmiaru.
pub fn laderman_multiplication(a: &Array2<f64>, b: &Array2<f64>, counter: &mut impl OperationCounter) -> Array2<f64> {
    let n = a.nrows();

    // 1. Baza rekurencji dla małych macierzy
    if n <= 9 {
        return binet_multiplication(a,b,counter)
    }

    // 2. Dopełnienie do wymiaru podzielnego przez 3
    let modulo_n = n % 3;
    let target_n;
    if modulo_n == 0{
        target_n = n;
    }else{
        if modulo_n == 1 {
            target_n = n+2;
        }else {
            target_n = n+1;
        }
    }
    // println!("target_n={}", target_n);
    let a_work;
    let b_work;
    if target_n != n {
        let mut a_padded = Array2::zeros((target_n, target_n));
        let mut b_padded = Array2::zeros((target_n, target_n));
        a_padded.slice_mut(s![0..n, 0..n]).assign(a);
        b_padded.slice_mut(s![0..n, 0..n]).assign(b);
        a_work = a_padded;
        b_work = b_padded;
    } else {
        a_work = a.to_owned();
        b_work = b.to_owned();
    }

    let size = target_n / 3;
    let s1 = size;
    let s2 = 2 * size;
    let t = target_n;

    // Podział macierzy roboczych na 9 bloków 3x3
    let a11 = a_work.slice(s![0..s1, 0..s1]);
    let a12 = a_work.slice(s![0..s1, s1..s2]);
    let a13 = a_work.slice(s![0..s1, s2..t]);
    let a21 = a_work.slice(s![s1..s2, 0..s1]);
    let a22 = a_work.slice(s![s1..s2, s1..s2]);
    let a23 = a_work.slice(s![s1..s2, s2..t]);
    let a31 = a_work.slice(s![s2..t, 0..s1]);
    let a32 = a_work.slice(s![s2..t, s1..s2]);
    let a33 = a_work.slice(s![s2..t, s2..t]);

    let b11 = b_work.slice(s![0..s1, 0..s1]);
    let b12 = b_work.slice(s![0..s1, s1..s2]);
    let b13 = b_work.slice(s![0..s1, s2..t]);
    let b21 = b_work.slice(s![s1..s2, 0..s1]);
    let b22 = b_work.slice(s![s1..s2, s1..s2]);
    let b23 = b_work.slice(s![s1..s2, s2..t]);
    let b31 = b_work.slice(s![s2..t, 0..s1]);
    let b32 = b_work.slice(s![s2..t, s1..s2]);
    let b33 = b_work.slice(s![s2..t, s2..t]);

    // 3. Obliczenie 23 iloczynów Ladermana rekurencyjnie
    let m1  = laderman_multiplication(&(a11.to_owned() + &a12 + &a13 - &a21 - &a22 - &a32 - &a33), &b22.to_owned(), counter);
    let m2  = laderman_multiplication(&(a11.to_owned() - &a21), &(-b12.to_owned() + &b22), counter);
    let m3  = laderman_multiplication(&a22.to_owned(), &(-b11.to_owned() + &b12 + &b21 - &b22 - &b23 - &b31 + &b33), counter);
    let m4  = laderman_multiplication(&(-a11.to_owned() + &a21 + &a22), &(b11.to_owned() - &b12 + &b22), counter);
    let m5  = laderman_multiplication(&(a21.to_owned() + &a22), &(-b11.to_owned() + &b12), counter);
    let m6  = laderman_multiplication(&a11.to_owned(), &b11.to_owned(), counter);
    let m7  = laderman_multiplication(&(-a11.to_owned() + &a31 + &a32), &(b11.to_owned() - &b13 + &b23), counter);
    let m8  = laderman_multiplication(&(-a11.to_owned() + &a31), &(b13.to_owned() - &b23), counter);
    let m9  = laderman_multiplication(&(a31.to_owned() + &a32), &(-b11.to_owned() + &b13), counter);
    let m10 = laderman_multiplication(&(a11.to_owned() + &a12 + &a13 - &a22 - &a23 - &a31 - &a32), &b23.to_owned(), counter);
    let m11 = laderman_multiplication(&a32.to_owned(), &(-b11.to_owned() + &b13 + &b21 - &b22 - &b23 - &b31 + &b32), counter);
    let m12 = laderman_multiplication(&(-a13.to_owned() + &a32 + &a33), &(b22.to_owned() + &b31 - &b32), counter);
    let m13 = laderman_multiplication(&(a13.to_owned() - &a33), &(b22.to_owned() - &b32), counter);
    let m14 = laderman_multiplication(&a13.to_owned(), &b31.to_owned(), counter);
    let m15 = laderman_multiplication(&(a32.to_owned() + &a33), &(-b31.to_owned() + &b32), counter);
    let m16 = laderman_multiplication(&(-a13.to_owned() + &a22 + &a23), &(b23.to_owned() + &b31 - &b33), counter);
    let m17 = laderman_multiplication(&(a13.to_owned() - &a23), &(b23.to_owned() - &b33), counter);
    let m18 = laderman_multiplication(&(a22.to_owned() + &a23), &(-b31.to_owned() + &b33), counter);
    let m19 = laderman_multiplication(&a12.to_owned(), &b21.to_owned(), counter);
    let m20 = laderman_multiplication(&a23.to_owned(), &b32.to_owned(), counter);
    let m21 = laderman_multiplication(&a21.to_owned(), &b13.to_owned(), counter);
    let m22 = laderman_multiplication(&a31.to_owned(), &b12.to_owned(), counter);
    let m23 = laderman_multiplication(&a33.to_owned(), &b33.to_owned(), counter);

    // 4. Składanie bloków wynikowych C (3x3 podmacierze wynikowe)
    let c11 = &m6 + &m14 + &m19;
    let c12 = &m1 + &m4 + &m5 + &m6 + &m12 + &m14 + &m15;
    let c13 = &m6 + &m7 + &m9 + &m10 + &m14 + &m16 + &m18;
    let c21 = &m2 + &m3 + &m4 + &m6 + &m14 + &m16 + &m17;
    let c22 = &m2 + &m4 + &m5 + &m6 + &m20;
    let c23 = &m14 + &m16 + &m17 + &m18 + &m21;
    let c31 = &m6 + &m7 + &m8 + &m11 + &m12 + &m13 + &m14;
    let c32 = &m12 + &m13 + &m14 + &m15 + &m22;
    let c33 = &m6 + &m7 + &m8 + &m9 + &m23;

    let mut c_work = Array2::zeros((target_n, target_n));
    c_work.slice_mut(s![0..s1, 0..s1]).assign(&c11);
    c_work.slice_mut(s![0..s1, s1..s2]).assign(&c12);
    c_work.slice_mut(s![0..s1, s2..t]).assign(&c13);
    c_work.slice_mut(s![s1..s2, 0..s1]).assign(&c21);
    c_work.slice_mut(s![s1..s2, s1..s2]).assign(&c22);
    c_work.slice_mut(s![s1..s2, s2..t]).assign(&c23);
    c_work.slice_mut(s![s2..t, 0..s1]).assign(&c31);
    c_work.slice_mut(s![s2..t, s1..s2]).assign(&c32);
    c_work.slice_mut(s![s2..t, s2..t]).assign(&c33);

    // 5. Obcięcie wyniku do oryginalnego rozmiaru n x n
    if target_n != n {
        c_work.slice(s![0..n, 0..n]).to_owned()
    } else {
        c_work
    }
}

pub fn dot_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a.dot(b)
}


#[cfg(test)]
mod tests {
    use assert_float_eq::assert_float_relative_eq;
    use ndarray::Zip;
    use crate::{Counter, EmptyCounter, generate_nxn_matrix};
    use super::*;
    #[test]
    fn test_binet_multiplication() {
        let n = 3;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = binet_multiplication(&m1, &m2, &mut EmptyCounter{});



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }

    #[test]
    fn test_binet_recursive_multiplication() {
        let n = 3;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = binet_recursive_multiplication(&m1, &m2, &mut EmptyCounter{});

        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }


    #[test]
    fn test_binet_multiplication_counter() {
        let n = 10;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let mut c = Counter::new();
        let _result = binet_multiplication(&m1, &m2,&mut c);
        let (additions, multiplications, divisions) = c.get_count();
        assert_eq!(n*n*(n-1), additions);
        assert_eq!(n*n*n, multiplications);
        assert_eq!(0, divisions);
    }
        
    #[test]
    fn test_strassen_multiplication_power_of_two() {
        let n = 64;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = strassen_multiplication(&m1, &m2, &mut EmptyCounter{});



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }   


    #[test]
    fn test_strassen_multiplication_not_power_of_two() {
        let n = 64+1;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = strassen_multiplication(&m1, &m2, &mut EmptyCounter{});



        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));

        assert!(result.shape() == &[n, n]);
    }


    #[test]
    fn test_strassen_multiplication_counter() {
    let n = 50;
    let m1 = generate_nxn_matrix(n);
    let m2 = generate_nxn_matrix(n);
    let mut c = Counter::new();
    let _result = strassen_multiplication(&m1, &m2, &mut c);
    let (additions, multiplications, divisions) = c.get_count();
    assert_eq!(131916, additions);
    assert_eq!(107653, multiplications);
    assert_eq!(0, divisions);
    } 


    #[test]
    fn test_laderman_multiplication_power_of_three() {
        let n = 81;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = laderman_multiplication(&m1, &m2, &mut EmptyCounter{});

        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));
        assert!(result.shape() == &[n, n]);
    }


    #[test]
    fn test_laderman_multiplication_not_power_of_three() {
        let n = 85;
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);
        let result = laderman_multiplication(&m1, &m2, &mut EmptyCounter{});

        let true_res = m1.dot(&m2);
        Zip::from(&result)
            .and(&true_res)
            .for_each(|a, b| assert_float_relative_eq!(*a, *b));
        assert!(result.shape() == &[n, n]);
    }



    #[test]
    fn test_laderman_multiplication_counter() {
    let n = 50;
    let m1 = generate_nxn_matrix(n);
    let m2 = generate_nxn_matrix(n);
    let mut c = Counter::new();
    let _result = laderman_multiplication(&m1, &m2, &mut c);
    let (additions, multiplications, divisions) = c.get_count();
    assert_eq!(95220, additions);
    assert_eq!(114264, multiplications);
    assert_eq!(0, divisions);
    } 

}