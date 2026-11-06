use std::time::Instant;

use z1::{generate_nxn_matrix,OperationCounter,Counter,binet_multiplication,strassen_multiplication,laderman_multiplication};
//use z1::EmptyCounter;

fn main() {

    let m = generate_nxn_matrix(10);
    let m2 = generate_nxn_matrix(10);


    let mut c1 = Counter::new();
    let mut c2 = Counter::new();
    let mut c3 = Counter::new();

    let start = Instant::now();
    let _res = binet_multiplication(&m, &m2, &mut c1);
    let t1 = start.elapsed();
    let _res2 = strassen_multiplication(&m, &m2, &mut c2);
    let t2 = start.elapsed()-t1;
    let _res3 = laderman_multiplication(&m, &m2, &mut c3);
    let t3 = start.elapsed()-t2;



    //println!("{:?}", res);
    println!("Counters: {:?}", c1.get_count());
    println!("Time taken for Binet multiplication: {:?}", t1);
    print!("\n");
    //println!("{:?}", res2);
    println!("Time taken for Strassen multiplication: {:?}", t2);
    println!("Counters: {:?}", c2.get_count());
    print!("\n");
    //println!("{:?}", res3);
    println!("Time taken for Laderman multiplication: {:?}", t3);
    println!("Counters: {:?}", c3.get_count());
}
