//use::z2::{invert_binet_strassen};
use z2::*;
use std::time::Instant;


fn main() {

    let a = generate_nxn_matrix(10000);
    //println!("Original matrix: \n{}", a);
    let mut counter = EmptyCounter::new();
    let start = Instant::now();
    let inv = recursive_matrix_inversion(&a, &mut counter, MultiplicationAlgorithm::Strassen, MatrixType::Normal).unwrap();
    let elapsed = start.elapsed();
    let inv2 = a.inv().unwrap();
    let elapsed2 = start.elapsed()-elapsed;
    println!("Time taken for Strassen inversion: {:?}", elapsed);
    println!("Time taken for library inversion: {:?}", elapsed2);
    //println!("Inverted matrix: \n{}", inv);


    //let check = binet_multiplication(&a, &inv, &mut counter);
    //println!("Check (A * A_inv): \n{}", check);

}

//Binet 1000
// debug 246s
// release 3.508
// RUSTFLAGS="-C target-cpu=native" 247
// RUSTFLAGS="-C target-cpu=native" + release 3.54
// RUSTFLAGS="-C target-cpu=x86-64" + release3.523
// Strassen



// 5000
// Strassen 130s
// Library 2.06


// 10000
// Strassen 918s
// Library 15,17s