use z2::*;

fn main() {
    let path = "z2/src/precision_comparison/";
    let seed: [u8; 32] = [0; 32];
    let mut rng = ChaCha8Rng::from_seed(seed);

    create_and_save_matrices(4, &mut rng, path);
    verify_all_factorizations(path);
}

fn create_and_save_matrices(n: usize, rng: &mut ChaCha8Rng, path: &str) {
    let matrix_a = generate_seeded_matrix(n, rng);

    println!("Macierz oryginalna A:\n{}", matrix_a);
    save_matrix_to_csv(&matrix_a, &format!("{}matrix_a.csv", path)).unwrap();

    let (binet_l, binet_u) =
        recursive_lu(&matrix_a, &mut EmptyCounter::new(), MultiplicationAlgorithm::Binet)
            .expect("Faktoryzacja LU metodą Bineta nie powiodła się");

    println!("\nFaktoryzacja Binet:");
    println!("L:\n{}", binet_l);
    println!("U:\n{}", binet_u);

    save_matrix_to_csv(
        &binet_l,
        &format!("{}lu/l_binet.csv", path),
    )
    .unwrap();
    save_matrix_to_csv(
        &binet_u,
        &format!("{}lu/u_binet.csv", path),
    )
    .unwrap();

    let (strassen_l, strassen_u) =
        recursive_lu(&matrix_a, &mut EmptyCounter::new(), MultiplicationAlgorithm::Strassen)
            .expect("Faktoryzacja LU metodą Strassena nie powiodła się");

    println!("\nFaktoryzacja Strassen:");
    println!("L:\n{}", strassen_l);
    println!("U:\n{}", strassen_u);

    save_matrix_to_csv(
        &strassen_l,
        &format!("{}lu/l_strassen.csv", path),
    )
    .unwrap();
    save_matrix_to_csv(
        &strassen_u,
        &format!("{}lu/u_strassen.csv", path),
    )
    .unwrap();
}

fn verify_all_factorizations(path: &str) {
    let matrix_a = load_matrix_from_csv(&format!("{}matrix_a.csv", path)).unwrap();

    let binet_l = load_matrix_from_csv(&format!("{}lu/l_binet.csv", path)).unwrap();
    let binet_u = load_matrix_from_csv(&format!("{}lu/u_binet.csv", path)).unwrap();

    let strassen_l =
        load_matrix_from_csv(&format!("{}lu/l_strassen.csv", path)).unwrap();
    let strassen_u =
        load_matrix_from_csv(&format!("{}lu/u_strassen.csv", path)).unwrap();

    verify_factorization("Binet", &binet_l, &binet_u, &matrix_a);
    verify_factorization("Strassen", &strassen_l, &strassen_u, &matrix_a);
}

fn verify_factorization(
    algorithm_name: &str,
    l: &Array2<f64>,
    u: &Array2<f64>,
    original_matrix: &Array2<f64>,
) {
    let reconstructed_matrix = l.dot(u);

    println!(
        "\nWeryfikacja faktoryzacji LU — {}: L * U porównane z A",
        algorithm_name
    );
    println!("Macierz odtworzona L * U:\n{}", reconstructed_matrix);
    println!("Macierz oryginalna A:\n{}", original_matrix);

    compare_matrices(&reconstructed_matrix, original_matrix);
}

fn compare_matrices(result: &Array2<f64>, expected: &Array2<f64>) {
    let difference = result - expected;

    let max_error = difference
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);

    let mean_absolute_error =
        difference.iter().map(|value| value.abs()).sum::<f64>()
            / difference.len() as f64;

    let frobenius_norm = difference
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();

    println!("--- Raport dokładności faktoryzacji ---");
    println!("Maksymalny błąd:      {:e}", max_error);
    println!("Średni błąd MAE:      {:e}", mean_absolute_error);
    println!("Norma Frobeniusa:     {:e}", frobenius_norm);
    println!("----------------------------------------");
}