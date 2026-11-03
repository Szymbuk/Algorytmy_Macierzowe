use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};

use z2::*;


fn benchmark_matrix_inversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("Matrix_inversion_comparison");
    
    group.sample_size(20);


    //let sizes = vec![16, 32, 64, 81, 128, 243, 256];
    let sizes = generate_sizes(1025 as usize);

    for n in sizes {

        let m1 = generate_nxn_matrix(n);
        let m2 = m1.clone();

        // Test 1: Binet
        group.bench_with_input(BenchmarkId::new("Binet", n), &n, |b, &_n| {

            b.iter(|| recursive_matrix_inversion(black_box(&m1), black_box(&mut EmptyCounter::new()), black_box(MultiplicationAlgorithm::Binet), black_box(MatrixType::Normal)))
        });

        // Test 2: Strassen
        group.bench_with_input(BenchmarkId::new("Strassen", n), &n, |b, &_n| {
            b.iter(|| recursive_matrix_inversion(black_box(&m1), black_box(&mut EmptyCounter::new()), black_box(MultiplicationAlgorithm::Strassen), black_box(MatrixType::Normal)))
        });

        // Test 3: Laderman
        group.bench_with_input(BenchmarkId::new("Laderman", n), &n, |b, &_n| {
            b.iter(|| recursive_matrix_inversion(black_box(&m1), black_box(&mut EmptyCounter::new()), black_box(MultiplicationAlgorithm::Laderman), black_box(MatrixType::Normal)))
        });


        // Test 4: Built-in inversion
        group.bench_with_input(BenchmarkId::new("Library_inversion", n), &n, |b, &_n| {
            b.iter(|| m2.inv().unwrap())
        });
    }

    group.finish();
}



fn benchmark_matrix_lu_factorization(c: &mut Criterion) {

    let mut group = c.benchmark_group("Matrix_lu_factorization_comparison");
    
    group.sample_size(20);


    //let sizes = vec![16, 32, 64, 81, 128, 243, 256];
    let sizes = generate_sizes(1025 as usize);

    for n in sizes {
        let m1 = generate_nxn_matrix(n);
        let m2 = m1.clone();

        // Test 1: Binet (Naiwny)
        group.bench_with_input(BenchmarkId::new("Binet", n), &n, |b, &_n| {
            // b.iter mierzy czas TYLKO tego, co wewnątrz
            // black_box zapobiega "zbyt sprytnemu" wycięciu kodu przez kompilator
            b.iter(|| recursive_lu(black_box(&m1), black_box(&mut EmptyCounter::new()), black_box(MultiplicationAlgorithm::Binet)))
        });

        // Test 2: Strassen
        group.bench_with_input(BenchmarkId::new("Strassen", n), &n, |b, &_n| {
            b.iter(|| recursive_lu(black_box(&m1), black_box(&mut EmptyCounter::new()), black_box(MultiplicationAlgorithm::Strassen)))
        });



        // Test 3: Built-in lu factorization
        group.bench_with_input(BenchmarkId::new("Library_lu_factorization", n), &n, |b, &_n| {
            b.iter(|| m2.factorize().unwrap())
        });
    }

    group.finish();
}


// Makra rejestrujące benchmark w systemie
criterion_group!(bench_inv, benchmark_matrix_inversion);
criterion_group!(bench_lu, benchmark_matrix_lu_factorization);
criterion_main!(bench_inv, bench_lu);
