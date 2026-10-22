use ndarray::Array2;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand_chacha::ChaCha8Rng;
use rand::prelude::*;
pub mod multiplication_algorithms;
pub mod csv_io;
pub use crate::multiplication_algorithms::{binet_multiplication, strassen_multiplication,laderman_multiplication,dot_product};



pub fn generate_nxn_matrix(n: usize) -> Array2<f64> {
    let matrix = Array2::random((n,n), Uniform::try_from(0.00000001.. 1.0).unwrap());
    matrix
}


pub fn generate_seeded_matrix(n: usize, rng: &mut ChaCha8Rng) -> ndarray::Array2<f64> {
    let mut matrix = ndarray::Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            matrix[[i, j]] = rng.random_range(0.00000001..1.0);
        }
    }
    matrix
}

pub fn generate_sizes(max_n: usize) -> Vec<usize> {
    let mut sizes = Vec::new();

    // 1. Baza: rzadszy krok liniowy dla ogólnego trendu (np. co 10 lub 20)
    for i in (10..=max_n).step_by(10) {
        sizes.push(i);
    }

    // 2. Wstrzyknięcie potęg 2 i punktów "zaraz obok" (dla Strassena)
    let powers_2 = [16, 32, 64, 128, 256, 512, 1024];
    for &p in &powers_2 {
        if p <= max_n {
            sizes.push(p - 1); // Zaraz przed (najgorszy przypadek paddingu)
            sizes.push(p);     // Idealna potęga (zero paddingu)
            sizes.push(p + 1); // Zaraz po (znowu duży padding)
        }
    }

    // 3. Wstrzyknięcie potęg 3 i punktów "zaraz obok" (dla Ladermana)
    let powers_3 = [27, 81, 243, 729];
    for &p in &powers_3 {
        if p <= max_n {
            sizes.push(p - 1);
            sizes.push(p);
            sizes.push(p + 1);
        }
    }

    // 4. Sortowanie i usunięcie duplikatów
    sizes.sort_unstable();
    sizes.dedup();

    sizes
}



pub trait OperationCounter {
    fn new() -> Self;
    fn increment_additions(&mut self, n: usize);
    fn increment_multiplications(&mut self, n: usize);
    fn increment_divisions(&mut self, n: usize);
    fn get_count(&self) -> (usize,usize,usize);
}

pub struct EmptyCounter {}

impl OperationCounter for EmptyCounter {
    fn new() -> Self {
        EmptyCounter {}
    }

    #[inline(always)]
    fn increment_additions(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_multiplications(&mut self, _n: usize) {}

    #[inline(always)]
    fn increment_divisions(&mut self, _n: usize) {}

    #[inline(always)] 
    fn get_count(&self) -> (usize,usize,usize) {
        (0,0,0)
    } 
}

pub struct Counter {
    count_additions: usize,
    count_multiplication: usize,
    count_division: usize,
}

impl OperationCounter for Counter {
    fn new() -> Self {
        Counter { count_additions: 0, count_multiplication: 0, count_division: 0 }
    }


    #[inline(always)]
    fn increment_additions(&mut self, n: usize) {
        self.count_additions += n;
    }

    #[inline(always)]
    fn increment_multiplications(&mut self, n: usize) {
        self.count_multiplication += n;
    }

    #[inline(always)]
    fn increment_divisions(&mut self, n: usize) {
        self.count_division += n;
    }

    #[inline(always)]
    fn get_count(&self) -> (usize,usize,usize) {
        (self.count_additions, self.count_multiplication, self.count_division)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BenchmarkData {
    pub n: usize,
    pub time_ms: f64,
    pub flops: u64,
}

pub fn calculate_empirical_complexity_exponent(data: &[BenchmarkData], use_flops: bool) -> f64 {
    let count = data.len() as f64;
    if count < 2.0 {
        return 0.0;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;

    for point in data {

        let x = (point.n as f64).ln();

        let y = if use_flops {
            (point.flops as f64).ln()
        } else {
            point.time_ms.ln()
        };

        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    // Wzór na współczynnik kierunkowy prostej (k) w regresji liniowej:
    // k = (N * Σ(xy) - Σx * Σy) / (N * Σ(x^2) - (Σx)^2)
    let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_x2 - sum_x * sum_x);

    slope
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use crate::multiplication_algorithms::binet_recursive_multiplication;
    use super::*;
    #[test]
    fn test_matrix_generation() {
        let n = 10;
        let m1 = generate_nxn_matrix(n);
        assert!(m1.shape() == &[n, n]);
        assert!(m1.iter().all(|x| *x >= 0.00000001 && *x < 1.0));
    }

    #[test]
    fn binet_complexity() {
        let sizes = vec![67,76,138,243,256,347,361,392,497,540];
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

    #[test]
    fn strassen_complexity() {
        let sizes = vec![67,76,138,243,256,347,361,392,497,540];
        // let sizes = vec![32, 64, 128, 256, 512];
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
    #[test]
    fn laderman_complexity() {
        let sizes = vec![67,76,138,243,256,347,361,392,497,540];
        // let sizes = vec![27, 81, 243, 729];
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
}