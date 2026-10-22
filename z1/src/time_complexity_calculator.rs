use z1::{binet_multiplication, strassen_multiplication, laderman_multiplication,generate_nxn_matrix, generate_sizes, Counter,OperationCounter,BenchmarkData, calculate_empirical_complexity_exponent};
use std::time::Instant;

fn main() {
    println!("=== TESTY ZŁOŻONOŚCI CZASOWEJ ALGORYTMÓW MNOŻENIA MACIERZY ===\n");

    binet_complexity();
    println!("\n");
    strassen_complexity();
    println!("\n");
    laderman_complexity();
}

fn binet_complexity() {
    //let sizes = generate_sizes(1025);
    //let sizes = vec![32, 64, 128, 256, 512,1024,2048];
    let sizes = vec![27, 81, 243, 729, 2187];
    // let sizes = vec![32, 64, 128, 256, 512];
    let mut binet_results = Vec::new();

    println!("--- ROZPOCZĘCIE TESTÓW DLA ALGORYTMU BINÉTA ---");

    for &n in &sizes {
        // Generowanie macierzy losowych A i B
        let a = generate_nxn_matrix(n);
        let b = generate_nxn_matrix(n);
        let mut counter = Counter::new();

        // Pomiar czasu
        let start = Instant::now();
        let _result = binet_multiplication(&a, &b, &mut counter);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let (add, mul, div) = counter.get_count();
        let data = BenchmarkData {
            n,
            time_ms: elapsed,
            flops: (add + mul + div) as u64,
        };

        println!("N = {:4} | Czas: {:8.2} ms | FLOPs: {:12}", n, elapsed, data.flops);
        binet_results.push(data);
    }

    // Wyznaczenie wykładników empirycznych
    let k_time = calculate_empirical_complexity_exponent(&binet_results, false);
    let k_flops = calculate_empirical_complexity_exponent(&binet_results, true);

    println!("\n=== WYNIKI DLA ALGORYTMU BINÉTA ===");
    println!("Praktyczna złożność (z czasu wykonania):    O(n^{:.2})", k_time);
    println!("Praktyczna złożność (z liczby operacji):   O(n^{:.2})", k_flops);
}


fn strassen_complexity() {
    //let sizes = generate_sizes(1025);
    let sizes = vec![27, 81, 243, 729, 2187];
    //let sizes = vec![32, 64, 128, 256, 512,1024,2048];
    let mut strassen_results = Vec::new();

    println!("--- ROZPOCZĘCIE TESTÓW DLA ALGORYTMU STRASSENA ---");

    for &n in &sizes {
        // Generowanie macierzy losowych A i B
        let a = generate_nxn_matrix(n);
        let b = generate_nxn_matrix(n);
        let mut counter = Counter::new();

        // Pomiar czasu
        let start = Instant::now();
        let _result = strassen_multiplication(&a, &b, &mut counter);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let (add, mul, div) = counter.get_count();
        let data = BenchmarkData {
            n,
            time_ms: elapsed,
            flops: (add + mul + div) as u64,
        };

        println!("N = {:4} | Czas: {:8.2} ms | FLOPs: {:12}", n, elapsed, data.flops);
        strassen_results.push(data);
    }

    // Wyznaczenie wykładników empirycznych
    let k_time = calculate_empirical_complexity_exponent(&strassen_results, false);
    let k_flops = calculate_empirical_complexity_exponent(&strassen_results, true);

    println!("\n=== WYNIKI DLA ALGORYTMU STRASSENA ===");
    println!("Praktyczna złożność (z czasu wykonania):    O(n^{:.2})", k_time);
    println!("Praktyczna złożność (z liczby operacji):   O(n^{:.2})", k_flops);
}

fn laderman_complexity() {
    //let sizes = generate_sizes(1025);
    let sizes = vec![27, 81, 243, 729, 2187];
    //let sizes = vec![32, 64, 128, 256, 512,1024,2048];
    let mut laderman_results = Vec::new();

    println!("--- ROZPOCZĘCIE TESTÓW DLA ALGORYTMU LADERMANA ---");

    for &n in &sizes {
        // Generowanie macierzy losowych A i B
        let a = generate_nxn_matrix(n);
        let b = generate_nxn_matrix(n);
        let mut counter = Counter::new();

        // Pomiar czasu
        let start = Instant::now();
        let _result = laderman_multiplication(&a, &b, &mut counter);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let (add, mul, div) = counter.get_count();
        let data = BenchmarkData {
            n,
            time_ms: elapsed,
            flops: (add + mul + div) as u64,
        };

        println!("N = {:4} | Czas: {:8.2} ms | FLOPs: {:12}", n, elapsed, data.flops);
        laderman_results.push(data);
    }

    // Wyznaczenie wykładników empirycznych
    let k_time = calculate_empirical_complexity_exponent(&laderman_results, false);
    let k_flops = calculate_empirical_complexity_exponent(&laderman_results, true);

    println!("\n=== WYNIKI DLA ALGORYTMU LADERMANA ===");
    println!("Praktyczna złożność (z czasu wykonania):    O(n^{:.2})", k_time);
    println!("Praktyczna złożność (z liczby operacji):   O(n^{:.2})", k_flops);
}