
use z2::*;

fn main(){
    let _path = "z2/src/precision_comparison/";
    let seed: [u8;32] = [0;32];  // constant seed for reproducibility
    let mut _rng = ChaCha8Rng::from_seed(seed);
    _create_and_save_matrices(4, &mut _rng, _path); // create and save matrices to CSV files
    compare_all_matrices(_path);
}


fn _create_and_save_matrices(n: usize, mut rng: &mut ChaCha8Rng, path: &str) {
    let matrix_a = generate_seeded_matrix(n, &mut rng);
    println!("Matrix:\n{}", matrix_a);
    save_matrix_to_csv(&matrix_a, &format!("{}{}", path, "matrix_a.csv")).unwrap();

    let  binet = recursive_matrix_inversion(&matrix_a, &mut EmptyCounter::new(), MultiplicationAlgorithm::Binet, MatrixType::Normal).unwrap();
    println!("Binet:\n{}", binet);
    save_matrix_to_csv(&binet, &format!("{}{}", path, "inversion/binet_inversion.csv")).unwrap();

    let  strassen = recursive_matrix_inversion(&matrix_a, &mut EmptyCounter::new(), MultiplicationAlgorithm::Strassen, MatrixType::Normal).unwrap();
    println!("Strassen:\n{}", strassen);
    save_matrix_to_csv(&strassen, &format!("{}{}", path, "inversion/strassen_inversion.csv")).unwrap();


    let  laderman = recursive_matrix_inversion_9block(&matrix_a, &mut EmptyCounter::new(), MultiplicationAlgorithm::Laderman).unwrap();
    println!("Laderman:\n{}", laderman);
    save_matrix_to_csv(&laderman, &format!("{}{}", path, "inversion/laderman_inversion.csv")).unwrap();


    let library_inversion = matrix_a.inv().unwrap();
    println!("Library inversion:\n{}", library_inversion);
    save_matrix_to_csv(&library_inversion, &format!("{}{}", path, "inversion/library_inv.csv")).unwrap();
}


fn compare_matrices(result: &Array2<f64>, ground_truth: &Array2<f64>) {
    // 1. Obliczamy macierz różnic (A - B)
    let difference = result - ground_truth;

    // 2. Maksymalny błąd bezwzględny
    let max_err = difference
        .iter()
        .map(|x| x.abs())
        .fold(0.0f64, f64::max); // Szuka największej wartości

    // 3. Średni błąd bezwzględny (MAE)
    let suma_bledow: f64 = difference.iter().map(|x| x.abs()).sum();
    let liczba_elementow = difference.len() as f64;
    let mae = suma_bledow / liczba_elementow;

    // 4. Norma Frobeniusa
    let frobenius: f64 = difference.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    // Wypisanie wyników z formatowaniem naukowym (wykładniczym)
    println!("--- Raport błędów zmiennoprzecinkowych ---");
    println!("Maksymalny błąd (Max Error) : {:e}", max_err);
    println!("Średni błąd (MAE)           : {:e}", mae);
    println!("Norma Frobeniusa            : {:e}", frobenius);
    println!("------------------------------------------");
}



fn compare_all_matrices(path: &str) {

    let binet = load_matrix_from_csv(&format!("{}{}", path, "inversion/binet_inversion.csv")).unwrap();
    let strassen = load_matrix_from_csv(&format!("{}{}", path, "inversion/strassen_inversion.csv")).unwrap();
    let laderman = load_matrix_from_csv(&format!("{}{}", path, "inversion/laderman_inversion.csv")).unwrap();
    let library_inversion = load_matrix_from_csv(&format!("{}{}", path, "inversion/library_inv.csv")).unwrap();



    println!("Porównanie Binet vs Library Inversion:");
    compare_matrices(&binet, &library_inversion);
    println!("\nPorównanie Strassen vs Library Inversion:");
    compare_matrices(&strassen, &library_inversion);
    println!("\nPorównanie Laderman vs Library Inversion:");
    compare_matrices(&laderman, &library_inversion);
}