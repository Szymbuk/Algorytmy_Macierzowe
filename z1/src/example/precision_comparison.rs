
use z1::{csv_io::save_matrix_to_csv,csv_io::load_matrix_from_csv,OperationCounter,EmptyCounter,binet_multiplication,strassen_multiplication,laderman_multiplication};
use ndarray::Array2;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn main(){
    let _path = "z1/src/example/";
    let seed: [u8;32] = [0;32];  // constant seed for reproducibility
    let mut _rng = ChaCha8Rng::from_seed(seed);
    //create_and_save_matrices(4, &mut _rng, _path); // create and save matrices to CSV files
    compare_all_matrices();
}


fn generate_seeded_matrix(n: usize, rng: &mut ChaCha8Rng) -> ndarray::Array2<f64> {
    let mut matrix = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] = rng.random_range(0.00000001..1.0);
        }
    }
    matrix
}


fn create_and_save_matrices(n: usize, mut rng: &mut ChaCha8Rng, path: &str) {
    let matrix_a = generate_seeded_matrix(n, &mut rng);
    println!("Matrix:\n{}", matrix_a);
    save_matrix_to_csv(&matrix_a, &format!("{}{}", path, "matrix_a.csv")).unwrap();

    let matrix_b = generate_seeded_matrix(n, &mut rng);
    println!("Matrix:\n{}", matrix_b);
    save_matrix_to_csv(&matrix_b, &format!("{}{}", path, "matrix_b.csv")).unwrap();



    let  binet = binet_multiplication(&matrix_a, &matrix_b, &mut EmptyCounter::new());
    println!("Binet:\n{}", binet);
    save_matrix_to_csv(&binet, &format!("{}{}", path, "binet.csv")).unwrap();

    let  strassen = strassen_multiplication(&matrix_a, &matrix_b, &mut EmptyCounter::new());
    println!("Strassen:\n{}", strassen);
    save_matrix_to_csv(&strassen, &format!("{}{}", path, "strassen.csv")).unwrap();


    let  laderman = laderman_multiplication(&matrix_a, &matrix_b, &mut EmptyCounter::new());
    println!("Laderman:\n{}", laderman);
    save_matrix_to_csv(&laderman, &format!("{}{}", path, "laderman.csv")).unwrap();


    let dot_product = matrix_a.dot(&matrix_b);
    println!("Dot product:\n{}", dot_product);
    save_matrix_to_csv(&dot_product, &format!("{}{}", path, "dot_product.csv")).unwrap();
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


fn compare_all_matrices() {
    let path = "z1/src/example/";

    let binet = load_matrix_from_csv(&format!("{}{}", path, "binet.csv")).unwrap();
    let strassen = load_matrix_from_csv(&format!("{}{}", path, "strassen.csv")).unwrap();
    let laderman = load_matrix_from_csv(&format!("{}{}", path, "laderman.csv")).unwrap();
    let dot_product = load_matrix_from_csv(&format!("{}{}", path, "dot_product.csv")).unwrap();
    let matlab = load_matrix_from_csv(&format!("{}{}", path, "matlab.csv")).unwrap();


    println!("Porównanie Binet vs Dot Product:");
    compare_matrices(&binet, &dot_product);
    println!("\nPorównanie Strassen vs Dot Product:");
    compare_matrices(&strassen, &dot_product);
    println!("\nPorównanie Laderman vs Dot Product:");
    compare_matrices(&laderman, &dot_product);
    println!("\nPorównanie Matlab vs Dot Product:");
    compare_matrices(&matlab, &dot_product);
}