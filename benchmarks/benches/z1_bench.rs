use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use z1::{EmptyCounter,OperationCounter, binet_multiplication, generate_nxn_matrix, generate_sizes, laderman_multiplication, strassen_multiplication,dot_product};

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix_multiplication_comparison");
    
    group.sample_size(20);

    //let sizes = vec![16, 32, 64, 81, 128, 243, 256];
    let sizes = generate_sizes(1 as usize);

    for n in sizes {
        let m1 = generate_nxn_matrix(n);
        let m2 = generate_nxn_matrix(n);

        // Test 1: Binet
        group.bench_with_input(BenchmarkId::new("Binet", n), &n, |b, &_n| {
            b.iter(|| binet_multiplication(black_box(&m1), black_box(&m2), black_box(&mut EmptyCounter::new())))
        });

        // Test 2: Strassen
        group.bench_with_input(BenchmarkId::new("Strassen", n), &n, |b, &_n| {
            b.iter(|| strassen_multiplication(black_box(&m1), black_box(&m2), black_box(&mut EmptyCounter::new())))
        });

        // Test 3: Laderman
        group.bench_with_input(BenchmarkId::new("Laderman", n), &n, |b, &_n| {
            b.iter(|| laderman_multiplication(black_box(&m1), black_box(&m2), black_box(&mut EmptyCounter::new())))
        });


        // Test 4: Built-in dot product
        group.bench_with_input(BenchmarkId::new("Dot-product", n), &n, |b, &_n| {
            b.iter(|| dot_product(black_box(&m1), black_box(&m2)))
        });
    }

    group.finish();
}

// Makra rejestrujące benchmark w systemie
criterion_group!(benches, benchmark_matrix_multiplication);
criterion_main!(benches);
