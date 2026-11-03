use crate::*;

pub fn recursive_matrix_inversion_9block<C: OperationCounter>(a: &Array2<f64>,
                                                              counter: &mut C,
                                                              m_algo: MultiplicationAlgorithm,
) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    if n == 1 {
        if a[[0, 0]] == 0.0 {
            return Err("Matrix is singular (division by zero)".to_string());
        }
        return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
    }

    // Jeśli wymiar nie jest podzielny przez 3, robimy padding do najbliższej wielokrotności 3
    let (a_work, is_padded, original_n) = make_divisible_by_3(a);
    let actual_n = a_work.nrows();

    let one_third = actual_n / 3;
    let two_thirds = one_third * 2;

    // Podział macierzy na 9 bloków (3x3 podmacierze)
    let a11 = a_work.slice(s![0..one_third, 0..one_third]).to_owned();
    let a12 = a_work.slice(s![0..one_third, one_third..two_thirds]).to_owned();
    let a13 = a_work.slice(s![0..one_third, two_thirds..actual_n]).to_owned();

    let a21 = a_work.slice(s![one_third..two_thirds, 0..one_third]).to_owned();
    let a22 = a_work.slice(s![one_third..two_thirds, one_third..two_thirds]).to_owned();
    let a23 = a_work.slice(s![one_third..two_thirds, two_thirds..actual_n]).to_owned();

    let a31 = a_work.slice(s![two_thirds..actual_n, 0..one_third]).to_owned();
    let a32 = a_work.slice(s![two_thirds..actual_n, one_third..two_thirds]).to_owned();
    let a33 = a_work.slice(s![two_thirds..actual_n, two_thirds..actual_n]).to_owned();

    // Pomocnicza domknięcie dla mnożenia metodą Ladermana
    // Utworzenie instancji mnożnika z algorytmem Ladermana
    let mut multiplier = MatrixMultiplier::new(counter, m_algo);

    // --- KROK 1: M1 = A11^-1 ---
    let m1 = recursive_matrix_inversion_9block(&a11, multiplier.counter, m_algo)?;

    // --- KROK 2: Redukcje pierwszego wiersza/kolumny ---
    let x21 = multiplier.multiply(&a21, &m1);
    let x31 = multiplier.multiply(&a31, &m1);
    let y12 = multiplier.multiply(&m1, &a12);
    let y13 = multiplier.multiply(&m1, &a13);

    // --- KROK 3: Dopełnienie Schura poziomu 1 (dla bloku 2x2) ---
    let x21_a12 = multiplier.multiply(&x21, &a12);
    let x21_a13 = multiplier.multiply(&x21, &a13);
    let x31_a12 = multiplier.multiply(&x31, &a12);
    let x31_a13 = multiplier.multiply(&x31, &a13);

    let s22 = &a22 - &x21_a12;
    let s23 = &a23 - &x21_a13;
    let s32 = &a32 - &x31_a12;
    let s33 = &a33 - &x31_a13;

    // --- KROK 4: Dopełnienie Schura poziomu 2 (redukcja S22) ---
    let s22_inv = recursive_matrix_inversion_9block(&s22, multiplier.counter, m_algo)?;

    let z32 = multiplier.multiply(&s32, &s22_inv);
    let w23 = multiplier.multiply(&s22_inv, &s23);

    let z32_s23 = multiplier.multiply(&z32, &s23);
    let s33_prime = &s33 - &z32_s23;

    let s33_prime_inv = recursive_matrix_inversion_9block(&s33_prime, multiplier.counter, m_algo)?;

    // --- KROK 5: Precyzyjna rekonstrukcja 9 bloków macierzy odwrotnej ---
    let b33 = s33_prime_inv;

    // B32 = - B33 * Z32
    let b33_z32 = multiplier.multiply(&b33, &z32);
    let b32 = &b33_z32 * -1.0;

    // B23 = - W23 * B33
    let w23_b33 = multiplier.multiply(&w23, &b33);
    let b23 = &w23_b33 * -1.0;

    // B22 = S22_inv + W23 * B33 * Z32  (= S22_inv - W23 * B32)
    let w23_b32 = multiplier.multiply(&w23, &b32);
    let b22 = &s22_inv - &w23_b32;

    // B31 = - B32 * X21 - B33 * X31
    let b32_x21 = multiplier.multiply(&b32, &x21);
    let b33_x31 = multiplier.multiply(&b33, &x31);
    let b31 = &(&b32_x21 + &b33_x31) * -1.0;

    // B21 = - B22 * X21 - B23 * X31
    let b22_x21 = multiplier.multiply(&b22, &x21);
    let b23_x31 = multiplier.multiply(&b23, &x31);
    let b21 = &(&b22_x21 + &b23_x31) * -1.0;

    // B13 = - Y12 * B23 - Y13 * B33
    let y12_b23 = multiplier.multiply(&y12, &b23);
    let y13_b33 = multiplier.multiply(&y13, &b33);
    let b13 = &(&y12_b23 + &y13_b33) * -1.0;

    // B12 = - Y12 * B22 - Y13 * B32
    let y12_b22 = multiplier.multiply(&y12, &b22);
    let y13_b32 = multiplier.multiply(&y13, &b32);
    let b12 = &(&y12_b22 + &y13_b32) * -1.0;

    // B11 = M1 - Y12 * B21 - Y13 * B31
    let y12_b21 = multiplier.multiply(&y12, &b21);
    let y13_b31 = multiplier.multiply(&y13, &b31);
    let b11 = &m1 - &(&y12_b21 + &y13_b31);

    // --- KROK 6: Złożenie macierzy wynikowej ---
    let mut c_work = Array2::zeros((actual_n, actual_n));

    c_work.slice_mut(s![0..one_third, 0..one_third]).assign(&b11);
    c_work.slice_mut(s![0..one_third, one_third..two_thirds]).assign(&b12);
    c_work.slice_mut(s![0..one_third, two_thirds..actual_n]).assign(&b13);

    c_work.slice_mut(s![one_third..two_thirds, 0..one_third]).assign(&b21);
    c_work.slice_mut(s![one_third..two_thirds, one_third..two_thirds]).assign(&b22);
    c_work.slice_mut(s![one_third..two_thirds, two_thirds..actual_n]).assign(&b23);

    c_work.slice_mut(s![two_thirds..actual_n, 0..one_third]).assign(&b31);
    c_work.slice_mut(s![two_thirds..actual_n, one_third..two_thirds]).assign(&b32);
    c_work.slice_mut(s![two_thirds..actual_n, two_thirds..actual_n]).assign(&b33);

    if is_padded {
        Ok(c_work.slice(s![0..original_n, 0..original_n]).to_owned())
    } else {
        Ok(c_work)
    }
}

#[cfg(test)]
mod tests_9block {
    use super::*;
    use ndarray::{Array2};

    #[test]
    fn test_recursive_matrix_inversion_9block_9x9() {
        // 1. Definicja przykładowej nieosobliwej macierzy 9x9 z dominującą przekątną
        let n = 6;
        let mut a = generate_nxn_matrix(n);
        let b = a.clone();

        let mut counter = EmptyCounter{};

        // 2. Wywołanie odwracania macierzy z podziałem na 9 bloków i Ladermanem
        let a_inv_9block = recursive_matrix_inversion_9block(&a, &mut counter, MultiplicationAlgorithm::Laderman)
            .expect("Odwracanie macierzy 9x9 metodą 9-blokową nie powiodło się");

        // 3. Obliczenie iloczynu A * A^(-1) przy użyciu domyślnego mnożenia
        let mut multiplier = MatrixMultiplier::new(&mut counter, MultiplicationAlgorithm::Laderman);
        let identity_result = multiplier.multiply(&a, &a_inv_9block);

        // 4. Weryfikacja 1: A * A^(-1) == I (Macierz jednostkowa 9x9)
        let expected_identity = Array2::<f64>::eye(n);
        let epsilon = 1e-5;

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (identity_result[[i, j]] - expected_identity[[i, j]]).abs() < epsilon,
                    "Błąd mnożenia A * A^(-1) na pozycji [{}, {}]: oczekiwano {}, otrzymano {}",
                    i, j, expected_identity[[i, j]], identity_result[[i, j]]
                );
            }
        }

        // 5. Weryfikacja 2 (Opcjonalna): Porównanie ze standardową metodą odwracania 2x2 (Binét)
        let b_inv_binet = recursive_matrix_inversion(&b, &mut counter, MultiplicationAlgorithm::Binet, MatrixType::Normal)
            .expect("Odwracanie macierzy metodą Binéta nie powiodło się");

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (a_inv_9block[[i, j]] - b_inv_binet[[i, j]]).abs() < epsilon,
                    "Rozbieżność między metodą 9-blokową a Binétem na pozycji [{}, {}]: 9block={}, Binet={}",
                    i, j, a_inv_9block[[i, j]], b_inv_binet[[i, j]]
                );
            }
        }

        println!("Test odwracania macierzy {}x{} z podziałem na 9 bloków zakończony sukcesem!",n,n);
    }
}