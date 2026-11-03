use crate::*;
pub fn recursive_matrix_inversion(a: &Array2<f64>,
                                  counter: &mut impl OperationCounter,
                                  m_algo: MultiplicationAlgorithm,
                                  m_type: MatrixType,
) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    // Baza rekurencji dla macierzy 1x1
    if n == 1 {
        if a[[0, 0]] == 0.0 {
            return Err("Matrix is singular (division by zero)".to_string());
        }
        return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
    }
    // Inicjalizacja pomocniczego pośrednika na początku funkcji
    let mut multiplier = MatrixMultiplier::new(counter, m_algo.clone());

    // Obsługa nieparzystego rozmiaru (padding)
    let (a_work, is_odd, original_n) = make_even(a);
    let actual_n = a_work.nrows();
    let mid = actual_n / 2;

    let a11 = a_work.slice(s![0..mid, 0..mid]);
    let a12 = a_work.slice(s![0..mid, mid..actual_n]);
    let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
    let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);

    let mut c_work = Array2::zeros((actual_n, actual_n));
    match m_type {
        MatrixType::Normal => {
            // === WARIANT 1: Macierz ogólna ===
            let a11_inv = recursive_matrix_inversion(&a11.to_owned(), multiplier.counter, m_algo, MatrixType::Normal)?;

            let a21_inv11 = multiplier.multiply(&a21.to_owned(), &a11_inv);
            
            let s22 = &a22 - multiplier.multiply(&a21_inv11, &a12.to_owned());
            let s22_inv = recursive_matrix_inversion(&s22, multiplier.counter, m_algo, MatrixType::Normal)?;

            let b22 = s22_inv.clone();
            let a11_inv_a12 = multiplier.multiply(&a11_inv, &a12.to_owned());
            let b12 = &multiplier.multiply(&a11_inv_a12, &s22_inv) * -1.0;

            let s22_inv_a21 = multiplier.multiply(&s22_inv, &a21.to_owned());
            let b21 = &multiplier.multiply(&s22_inv_a21, &a11_inv) * -1.0;

            let i_matrix = Array2::eye(mid);
            let a12_s22 = multiplier.multiply(&a12.to_owned(), &s22_inv);
            let a12_s22_a21 = multiplier.multiply(&a12_s22, &a21.to_owned());
            let a12_s22_a21_a11_inv = multiplier.multiply(&a12_s22_a21, &a11_inv);
            let b11 = multiplier.multiply(&a11_inv, &(&i_matrix + &a12_s22_a21_a11_inv));

            c_work.slice_mut(s![0..mid, 0..mid]).assign(&b11);
            c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
            c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
            c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&b22);

            multiplier.counter.increment_additions(mid * mid * 3); // 2 dodawania/odejmowania macierzy + 1 dodawanie macierzy I
            multiplier.counter.increment_multiplications(mid * mid * 2); // 2 zmiany znaku (* -1.0) dla b12 i b21
        }

        MatrixType::Lower => {
            /*
            === WARIANT 2: Macierz dolnotrójkątna ===
            Odwrotność macierzy dolnotrójkątnej:
            [ A11    0  ]^-1 = [ A11_inv                      0    ]
            [ A21   A22 ]      [ -A22_inv * A21 * A11_inv  A22_inv ]
            */

            let a11_inv = recursive_matrix_inversion(&a11.to_owned(), multiplier.counter, m_algo, MatrixType::Lower)?;
            let a22_inv = recursive_matrix_inversion(&a22.to_owned(), multiplier.counter, m_algo, MatrixType::Lower)?;
            // B21 = - A22_inv * A21 * A11_inv
            let temp = multiplier.multiply(&a21.to_owned(), &a11_inv);
            let b21 = &multiplier.multiply(&a22_inv, &temp) * -1.0;

            c_work.slice_mut(s![0..mid, 0..mid]).assign(&a11_inv);
            // Górny prawy blok (pomiędzy 0..mid a mid..actual_n) pozostaje zerowy (zainicjalizowany jako 0)
            c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
            c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&a22_inv);
            
            multiplier.counter.increment_multiplications(mid * mid); // 1 zmiana znaku (* -1.0) dla b21
        }

        MatrixType::Upper => {
            /*
            === WARIANT 3: Macierz górnotrójkątna ===
            Odwrotność macierzy górnotrójkątnej:
            [ A11  A12 ]^-1 = [ A11_inv  -A11_inv * A12 * A22_inv ]
            [  0   A22 ]      [    0             A22_inv          ]
            */

            let a11_inv = recursive_matrix_inversion(&a11.to_owned(), multiplier.counter, m_algo, MatrixType::Upper)?;
            let a22_inv = recursive_matrix_inversion(&a22.to_owned(), multiplier.counter, m_algo, MatrixType::Upper)?;
            // B12 = - A11_inv * A12 * A22_inv
            let temp = multiplier.multiply(&a12.to_owned(), &a22_inv);
            let b12 = &multiplier.multiply(&a11_inv, &temp) * -1.0;

            c_work.slice_mut(s![0..mid, 0..mid]).assign(&a11_inv);
            c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
            // Dolny lewy blok pozostaje zerowy
            c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&a22_inv);

            multiplier.counter.increment_multiplications(mid * mid); // 1 zmiana znaku (* -1.0) dla b12
        }
    }

    if is_odd {
        Ok(c_work.slice(s![0..original_n, 0..original_n]).to_owned())
    } else {
        Ok(c_work)
    }
    // // 1. Rekurencyjne odwrócenie lewego górnego bloku: inverse(A11)
    // // Tworzymy na chwilę pod-pośrednika lub przekazujemy licznik w głąb rekurencji
    // let a11_inv = recursive_matrix_inversion(&a11.to_owned(), multiplier.counter, m_algo, m_type)?;
    //
    // // 2. Obliczenie dopełnienia Schura: S22 = A22 - A21 * inverse(A11) * A12
    // let a21_inv11 = multiplier.multiply(&a21.to_owned(), &a11_inv);
    // let m1 = multiplier.multiply(&a21_inv11, &a12.to_owned());
    // let s22 = &a22 - &m1;
    //
    // // 3. Rekurencyjne odwrócenie dopełnienia Schura: S22_inv = inverse(S22)
    // let s22_inv = recursive_matrix_inversion(&s22, multiplier.counter, m_algo, m_type)?;
    //
    // // 4. Obliczenie 4 bloków macierzy odwrotnej za pomocą czystego interfejsu .multiply():
    // let b22 = s22_inv.clone();
    //
    // // B12 = - inverse(A11) * A12 * S22_inv
    // let a11_inv_a12 = multiplier.multiply(&a11_inv, &a12.to_owned());
    // let b12 = &multiplier.multiply(&a11_inv_a12, &s22_inv) * -1.0;
    //
    // // B21 = - S22_inv * A21 * inverse(A11)
    // let s22_inv_a21 = multiplier.multiply(&s22_inv, &a21.to_owned());
    // let b21 = &multiplier.multiply(&s22_inv_a21, &a11_inv) * -1.0;
    //
    // // B11 = inverse(A11) * (I + A12 * S22_inv * A21 * inverse(A11))
    // let i_matrix = Array2::eye(mid);
    // let a12_s22 = multiplier.multiply(&a12.to_owned(), &s22_inv);
    // let a12_s22_a21 = multiplier.multiply(&a12_s22, &a21.to_owned());
    // let a12_s22_a21_a11_inv = multiplier.multiply(&a12_s22_a21, &a11_inv);
    // let bracket = &i_matrix + &a12_s22_a21_a11_inv;
    // let b11 = multiplier.multiply(&a11_inv, &bracket);
    //
    // // 5. Scalenie bloków i ewentualne usunięcie paddingu
    // let mut c_work = Array2::zeros((actual_n, actual_n));
    // c_work.slice_mut(s![0..mid, 0..mid]).assign(&b11);
    // c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
    // c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
    // c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&b22);
    //
    // if is_odd {
    //     Ok(c_work.slice(s![0..original_n, 0..original_n]).to_owned())
    // } else {
    //     Ok(c_work)
    // }

    // ===========================================================================
    // Wersja bez pomocniczej klasy

    // // 1. Rekurencyjne odwrócenie lewego górnego bloku: inverse(A11)
    // let a11_inv = recursive_matrix_inversion(&a11.to_owned(), counter, m_algo)?;
    //
    // // 2. Obliczenie dopełnienia Schura: S22 = A22 - A21 * inverse(A11) * A12
    // let a21_inv11 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a21.to_owned(), &a11_inv, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a21.to_owned(), &a11_inv, counter),
    // };
    //
    // let m1 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a21_inv11, &a12.to_owned(), counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a21_inv11, &a12.to_owned(), counter),
    // };
    // let s22 = &a22 - &m1;
    //
    // // 3. Rekurencyjne odwrócenie dopełnienia Schura: S22_inv = inverse(S22)
    // let s22_inv = recursive_matrix_inversion(&s22, counter, m_algo)?;
    //
    // // 4. Wyznaczenie 4 bloków macierzy odwrotnej (B11, B12, B21, B22)
    // let b22 = s22_inv.clone();
    //
    // // B12 = - inverse(A11) * A12 * S22_inv
    // let a11_inv_a12 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a11_inv, &a12.to_owned(), counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a11_inv, &a12.to_owned(), counter),
    // };
    // let b12 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a11_inv_a12, &s22_inv, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a11_inv_a12, &s22_inv, counter),
    // } * -1.0;
    //
    // // B21 = - S22_inv * A21 * inverse(A11)
    // let s22_inv_a21 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&s22_inv, &a21.to_owned(), counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&s22_inv, &a21.to_owned(), counter),
    // };
    // let b21 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&s22_inv_a21, &a11_inv, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&s22_inv_a21, &a11_inv, counter),
    // } * -1.0;
    //
    // // B11 = inverse(A11) * (I + A12 * S22_inv * A21 * inverse(A11))
    // let i_matrix = Array2::eye(mid);
    // let a12_s22 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a12.to_owned(), &s22_inv, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a12.to_owned(), &s22_inv, counter),
    // };
    // let a12_s22_a21 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a12_s22, &a21.to_owned(), counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a12_s22, &a21.to_owned(), counter),
    // };
    // let a12_s22_a21_a11_inv = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a12_s22_a21, &a11_inv, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a12_s22_a21, &a11_inv, counter),
    // };
    // let bracket = &i_matrix + &a12_s22_a21_a11_inv;
    //
    // let b11 = match m_algo {
    //     MultiplicationAlgorithm::Binet => binet_multiplication(&a11_inv, &bracket, counter),
    //     MultiplicationAlgorithm::Strassen => strassen_multiplication(&a11_inv, &bracket, counter),
    // };
    //
    // // 5. Składanie bloków w jedną macierz roboczą
    // let mut c_work = Array2::zeros((actual_n, actual_n));
    // c_work.slice_mut(s![0..mid, 0..mid]).assign(&b11);
    // c_work.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
    // c_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
    // c_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&b22);
    //
    // // 6. Usunięcie paddingu, jeśli oryginalny rozmiar był nieparzysty
    // if is_odd {
    //     Ok(c_work.slice(s![0..original_n, 0..original_n]).to_owned())
    // } else {
    //     Ok(c_work)
    // }
}

// pub fn matrix_invert_binet(a: &Array2<f64>, counter: &mut impl OperationCounter) -> Result<Array2<f64>, String> {
//     let n = a.nrows();
//     if n != a.ncols() {
//         return Err("Matrix must be square".to_string());
//     }
//
//     if n == 1 {
//         return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
//     }
//
//     // 2. Obsługa nieparzystego wymiaru (dynamiczne powiększenie do parzystego)
//     let (a_work, is_odd, _original_n) = make_even(&a);
//
//     let actual_n = a_work.nrows();
//
//     let mid = actual_n / 2;
//
//     // Podział macierzy roboczych na 4 równe bloki
//     let a11 = a_work.slice(s![0..mid, 0..mid]);
//     let a12 = a_work.slice(s![0..mid, mid..actual_n]);
//     let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
//     let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);
//
//     // inverse(a11)
//     let a11_inv = matrix_invert_binet(&a11.to_owned(), counter)?;
//     // S22 = A22 - A21 * inverse(A11) * A12
//     let s22 = &a22 - binet_multiplication(&(binet_multiplication(&(a21.to_owned()),&a11_inv, counter).to_owned()),&(a12.to_owned()), counter);
//     // S22_inv = inverse(S22)
//     let s22_inv = matrix_invert_binet(&s22, counter)?;
//     // B11 = inverse(A11) * (I + A12 * S22_inv * A21 * inverse(A11))
//     let i = Array2::eye(a11_inv.nrows());
//     let a12_s22 = binet_multiplication(&(a12.to_owned()), &s22_inv, counter);
//     let a12_s22_a21 = binet_multiplication(&(a12_s22.to_owned()), &(a21.to_owned()), counter);
//     let a12_s22_a21_a11_inv = binet_multiplication(&(a12_s22_a21.to_owned()), &(a11_inv.to_owned()), counter);
//     let b11 = binet_multiplication(&a11_inv, &(i + a12_s22_a21_a11_inv), counter);
//
//     // B12 = - inverse(A11) * A12 * inverse(S22)
//     let b12 = -binet_multiplication(&(binet_multiplication(&(a11_inv.to_owned()), &(a12.to_owned()), counter)), &s22_inv, counter);
//     // B21 = - S22_inv * A21 * inverse(A11)
//     let b21 = -binet_multiplication(&(binet_multiplication(&(s22_inv.to_owned()), &(a21.to_owned()), counter)), &a11_inv, counter);
//     // B22 = S22_inv
//     let b22 = s22_inv;
//     // // M6 = (A21 - A11) * (B11 + B12)
//     // let m6 = strassen_multiplication(&(a21.to_owned() - &a11), &(b11.to_owned() + &b12),counter);
//     // // M7 = (A12 - A22) * (B21 + B22)
//     // let m7 = strassen_multiplication(&(a12.to_owned() - &a22), &(b21.to_owned() + &b22),counter);
//
//     // // Składanie bloków wynikowych C
//     // let c11 = &m1 + &m4 - &m5 + &m7;
//     // let c12 = &m3 + &m5;
//     // let c21 = &m2 + &m4;
//     // let c22 = &m1 - &m2 + &m3 + &m6;
//
//     counter.increment_additions(a11.nrows() * a11.ncols() * 18);
//
//     let mut a_inv = Array2::zeros((actual_n, actual_n));
//     a_inv.slice_mut(s![0..mid, 0..mid]).assign(&b11);
//     a_inv.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
//     a_inv.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
//     a_inv.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&b22);
//
//     // 4. Obcięcie wyniku do oryginalnego rozmiaru, jeśli wymiar wejściowy był nieparzysty
//     if is_odd {
//         Ok(a_inv.slice(s![0..n, 0..n]).to_owned())
//     } else {
//         Ok(a_inv)
//     }
//
//
// }


// pub fn matrix_invert_strassen(a: Array2<f64>, counter: &mut impl OperationCounter) -> Result<Array2<f64>, String> {
//     let n = a.nrows();
//     if n != a.ncols() {
//         return Err("Matrix must be square".to_string());
//     }

//     if n == 1 {
//         return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
//     }

//     let mid = n / 2;
//     let a11 = a.slice(s![..mid, ..mid]).to_owned();
//     let a12 = a.slice(s![..mid, mid..]).to_owned();
//     let a21 = a.slice(s![mid.., ..mid]).to_owned();
//     let a22 = a.slice(s![mid.., mid..]).to_owned();

//     let inv_a11 = matrix_invert_strassen(a11, counter)?;
//     let inv_a22 = matrix_invert_strassen(a22, counter)?;

//     let s1 = &a21.dot(&inv_a11);
//     let s2 = &inv_a22.dot(&a12);
//     let s3 = &s1.dot(&s2);

//     let inv_a11_s3 = &inv_a11 - &s3;

//     let mut result = Array2::<f64>::zeros((n, n));
//     result.slice_mut(s![..mid, ..mid]).assign(&inv_a11_s3);
//     result.slice_mut(s![..mid, mid..]).assign(&(-&inv_a11.dot(&a12).dot(&inv_a22)));
//     result.slice_mut(s![mid.., ..mid]).assign(&(-&inv_a22.dot(&a21).dot(&inv_a11)));
//     result.slice_mut(s![mid.., mid..]).assign(&inv_a22);

//     Ok(result)
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_matrix_inversion_binet_and_strassen() {
        // 1. Zdefiniowanie przykładowej macierzy kwadratowej A 4x4
        // let a = array![
        //     [4.0, 7.0, 2.0, 3.0],
        //     [0.0, 2.0, 1.0, 5.0],
        //     [1.0, 0.0, 3.0, 2.0],
        //     [2.0, 4.0, 1.0, 1.0]
        // ];
        let a = generate_nxn_matrix(9);

        let mut counter = EmptyCounter {};

        // 2. Test dla algorytmu Binéta
        let a_inv_binet = recursive_matrix_inversion(&a, &mut counter, MultiplicationAlgorithm::Binet, MatrixType::Normal)
            .expect("Odwracanie macierzy metodą Binéta nie powiodło się");

        // Obliczenie A * A^(-1)
        let identity_binet = binet_multiplication(&a, &a_inv_binet, &mut counter);

        // 3. Test dla algorytmu Strassena
        let a_inv_strassen = recursive_matrix_inversion(&a, &mut counter, MultiplicationAlgorithm::Strassen, MatrixType::Normal)
            .expect("Odwracanie macierzy metodą Strassena nie powiodło się");

        // Obliczenie A * A^(-1)
        let identity_strassen = strassen_multiplication(&a, &a_inv_strassen, &mut counter);

        // 4. Weryfikacja: A * A^(-1) powinno być bliskie macierzy jednostkowej I
        let expected_identity = Array2::<f64>::eye(4);
        let epsilon = 1e-6; // Tolerancja dla błędów zaokrągleń zmiennoprzecinkowych

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (identity_binet[[i, j]] - expected_identity[[i, j]]).abs() < epsilon,
                    "Błąd Binéta na pozycji [{}, {}]: oczekiwano {}, otrzymano {}",
                    i, j, expected_identity[[i, j]], identity_binet[[i, j]]
                );

                assert!(
                    (identity_strassen[[i, j]] - expected_identity[[i, j]]).abs() < epsilon,
                    "Błąd Strassena na pozycji [{}, {}]: oczekiwano {}, otrzymano {}",
                    i, j, expected_identity[[i, j]], identity_strassen[[i, j]]
                );
            }
        }

        println!("Test odwracania macierzy zakończony sukcesem dla obu metod!");
    }
}