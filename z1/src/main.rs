use z1::{generate_nxn_matrix,OperationCounter,Counter,EmptyCounter,binet_multiplication,strassen_multiplication,laderman_multiplication};


fn main() {

    let m = generate_nxn_matrix(10);
    let m2 = generate_nxn_matrix(10);


    let mut c1 = Counter::new();
    let mut c2 = Counter::new();
    let mut c3 = Counter::new();

    let res = binet_multiplication(&m, &m2, &mut c1);
    let res2 = strassen_multiplication(&m, &m2, &mut c2);
    let res3 = laderman_multiplication(&m, &m2, &mut c3);

    println!("{:?}", res);
    println!("Counters: {:?}", c1.get_count());
    print!("\n");
    println!("{:?}", res2);
    println!("Counters: {:?}", c2.get_count());
    print!("\n");
    println!("{:?}", res3);
    println!("Counters: {:?}", c3.get_count());
}
