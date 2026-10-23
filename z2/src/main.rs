//use::z2::{invert_binet_strassen};
use z2::*;



fn main() {
    println!("Hello, world!");
    let a = generate_nxn_matrix(4);
    println!("Original matrix: \n{}", a);
    let mut counter = EmptyCounter::new();
    let inv = matrix_invert_binet(&a, &mut counter).unwrap();

    println!("Inverted matrix: \n{}", inv);


    let check = binet_multiplication(&a, &inv, &mut counter);
    println!("Check (A * A_inv): \n{}", check);

}
