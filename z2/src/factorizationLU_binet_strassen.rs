use crate::*;

pub fn recursive_lu<C : OperationCounter>(a: &Array2<f64>, counter: &mut C, m_algo: MultiplicationAlgorithm, ) -> Result<(Array2<f64>, Array2<f64>), String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    // Baza rekurencji dla macierzy 1x1
    if n == 1 {
        let val = a[[0, 0]];
        if val == 0.0 {
            // W pełnym rozkładu LU bez pivotingu 0 na przekątnej oznacza awarię
            return Err("Zero pivot encountered (matrix singular)".to_string());
        }
        // Dla 1x1: L = [[1.0]], U = [[a00]]
        let l = Array2::from_elem((1, 1), 1.0);
        let u = Array2::from_elem((1, 1), val);
        return Ok((l, u));
    }

    let mut multiplier = MatrixMultiplier::new(counter, m_algo);

    // Obsługa nieparzystego wymiaru (padding zerami, a dla L na przekątnej jedynkami)
    let (a_work, is_odd, original_n) = make_even(a);
    let actual_n = a_work.nrows();
    let mid = actual_n / 2;

    let a11 = a_work.slice(s![0..mid, 0..mid]);
    let a12 = a_work.slice(s![0..mid, mid..actual_n]);
    let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
    let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);

    // 1. Rekurencyjna faktoryzacja LU lewego górnego bloku: A11 = L11 * U11
    let (l11, u11) = recursive_lu(&a11.to_owned(), multiplier.counter, m_algo)?;

    // 2. Wyznaczenie U12: L11 * U12 = A12  =>  U12 = L11_inv * A12
    // Używamy naszego odwracania dla macierzy trójkątnej dolnej!
    let l11_inv = recursive_matrix_inversion(&l11, multiplier.counter, m_algo, MatrixType::Lower)?;
    let u12 = multiplier.multiply(&l11_inv, &a12.to_owned());

    // 3. Wyznaczenie L21: L21 * U11 = A21  =>  L21 = A21 * U11_inv
    // Używamy odwracania dla macierzy trójkątnej górnej!
    let u11_inv = recursive_matrix_inversion(&u11, multiplier.counter, m_algo, MatrixType::Upper)?;
    let l21 = multiplier.multiply(&a21.to_owned(), &u11_inv);

    // 4. Obliczenie dopełnienia Schura: A22' = A22 - L21 * U12
    let l21_u12 = multiplier.multiply(&l21, &u12);
    let a22_schur = &a22 - &l21_u12;

    // 5. Rekurencyjna faktoryzacja LU dopełnienia Schura: A22' = L22 * U22
    let (l22, u22) = recursive_lu(&a22_schur, multiplier.counter, m_algo)?;

    // 6. Składanie pełnych macierzy wynikowych L oraz U
    let mut l_work = Array2::eye(actual_n); // L ma jedynki na głównej przekątnej
    let mut u_work = Array2::zeros((actual_n, actual_n));

    l_work.slice_mut(s![0..mid, 0..mid]).assign(&l11);
    l_work.slice_mut(s![mid..actual_n, 0..mid]).assign(&l21);
    l_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&l22);

    u_work.slice_mut(s![0..mid, 0..mid]).assign(&u11);
    u_work.slice_mut(s![0..mid, mid..actual_n]).assign(&u12);
    u_work.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&u22);

    // 7. Usunięcie paddingu, jeśli oryginalny rozmiar był nieparzysty
    if is_odd {
        let l_final = l_work.slice(s![0..original_n, 0..original_n]).to_owned();
        let u_final = u_work.slice(s![0..original_n, 0..original_n]).to_owned();
        Ok((l_final, u_final))
    } else {
        Ok((l_work, u_work))
    }
}

#[cfg(test)]
mod lu_tests {
    use super::*;

    #[test]
    fn test_recursive_lu_decomposition() {
        // 1. Generowanie przykładowej macierzy 4x4
        let a = generate_nxn_matrix(4);

        let mut counter = EmptyCounter{};

        // 2. Wywołanie faktoryzacji LU z użyciem np. metody Binéta
        let (l, u) = recursive_lu(&a, &mut counter, MultiplicationAlgorithm::Binet).expect("Faktoryzacja LU nie powiodła się");

        // 3. Weryfikacja 1: Sprawdzenie, czy L * U == A
        // Do mnożenia L i U używamy naszego sprawdzonego mnożnika
        let mut multiplier = MatrixMultiplier::new(&mut counter, MultiplicationAlgorithm::Binet);
        let reconstructed_a = multiplier.multiply(&l, &u);

        let epsilon = 1e-5;
        let n = a.nrows();

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (reconstructed_a[[i, j]] - a[[i, j]]).abs() < epsilon,
                    "Błąd rekonstrukcji L*U na pozycji [{}, {}]: oczekiwano {}, otrzymano {}", i, j, a[[i, j]], reconstructed_a[[i, j]]
                );
            }
        }

        // 4. Weryfikacja 2: Sprawdzenie własności trójkątności macierzy L i U
        for i in 0..n {
            for j in 0..n {
                if i > j {
                    // Poniżej głównej przekątnej w U muszą być same zera
                    assert!(
                        u[[i, j]].abs() < epsilon, "Macierz U nie jest górnotrójkątna na pozycji [{}, {}] (wartość: {})", i, j, u[[i, j]]
                    );
                } else if i < j {
                    // Powyżej głównej przekątnej w L muszą być same zera
                    assert!(
                        l[[i, j]].abs() < epsilon, "Macierz L nie jest dolnotrójkątna na pozycji [{}, {}] (wartość: {})", i, j, l[[i, j]]
                    );
                } else {
                    // Na głównej przekątnej L muszą być jedynki
                    assert!(
                        (l[[i, j]] - 1.0).abs() < epsilon, "Przekątna macierzy L musi zawierać jedynki na pozycji [{}, {}]", i, j
                    );
                }
            }
        }

        println!("Test rozkładu LU zakończony pełnym sukcesem!");
    }
}