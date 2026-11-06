use z2::*;

fn main(){
    let multiplication_types = vec![MultiplicationAlgorithm::Binet, MultiplicationAlgorithm::Strassen];
    let function_names = vec!["Binet", "Strassen"];
    for (mult, name) in multiplication_types.iter().zip(function_names.iter()) {
        println!("Testing inversion function: {}", name);
        let sizes = generate_sizes(1024 as usize);
        let results = measure_operation_count_binet_strassen(*mult, sizes.clone());
        println!();
        let mut wtr = Writer::from_path(format!("z2/src/operation_comparison/{name}_inversion_operation_count.csv")).unwrap();
        wtr.write_record(&["Size", "Additions", "Multiplications"]).unwrap();
        for (size, (add, mult)) in sizes.iter().zip(results.iter()) {
            wtr.write_record(&[size.to_string(), add.to_string(), mult.to_string()]).unwrap();
        }

        wtr.flush().unwrap();
    } 


    let name = "Laderman";
    println!("Testing inversion function: {}", name);
    let sizes = generate_sizes(1024 as usize);
    let results = measure_operation_count_laderman(sizes.clone());
    println!();
    let mut wtr = Writer::from_path(format!("z2/src/operation_comparison/{name}_inversion_operation_count.csv")).unwrap();
    wtr.write_record(&["Size", "Additions", "Multiplications"]).unwrap();
    for (size, (add, mult)) in sizes.iter().zip(results.iter()) {
        wtr.write_record(&[size.to_string(), add.to_string(), mult.to_string()]).unwrap();
    }
    wtr.flush().unwrap();

        
    let multiplication_types = vec![MultiplicationAlgorithm::Binet, MultiplicationAlgorithm::Strassen];
    let function_names = vec!["Binet", "Strassen"];
    for (mult, name) in multiplication_types.iter().zip(function_names.iter()) {
        println!("Testing lu factorization function: {}", name);
        let sizes = generate_sizes(1024 as usize);
        let results = measure_operation_count_lu_binet_strassen(*mult, sizes.clone());
        println!();
        let mut wtr = Writer::from_path(format!("z2/src/operation_comparison/{name}_lu_operation_count.csv")).unwrap();
        wtr.write_record(&["Size", "Additions", "Multiplications"]).unwrap();
        for (size, (add, mult)) in sizes.iter().zip(results.iter()) {
            wtr.write_record(&[size.to_string(), add.to_string(), mult.to_string()]).unwrap();
        }

        wtr.flush().unwrap();
    } 

    //landerman_strassen_optimalization_test();
}




fn measure_operation_count_binet_strassen(m_algo: MultiplicationAlgorithm, sizes: Vec<usize>) -> Vec<(usize, usize)> {
    let mut results = vec![];
    for &size in sizes.iter() {
        let matrix_a = generate_nxn_matrix(size);

        let mut counter = Counter::new();
        let _result = recursive_matrix_inversion(&matrix_a,  &mut counter,m_algo, MatrixType::Normal);

        let (additions, multiplications, _divisions) = counter.get_count();
        println!("Size: {}, Additions: {}, Multiplications: {}", size, additions, multiplications);
        results.push((additions, multiplications));
    }
    results
}

fn measure_operation_count_laderman(sizes: Vec<usize>) -> Vec<(usize, usize)> {
    let mut results = vec![];
    for &size in sizes.iter() {
        let matrix_a = generate_nxn_matrix(size);

        let mut counter = Counter::new();
        let _result = recursive_matrix_inversion_9block(&matrix_a,  &mut counter, MultiplicationAlgorithm::Laderman);

        let (additions, multiplications, _divisions) = counter.get_count();
        println!("Size: {}, Additions: {}, Multiplications: {}", size, additions, multiplications);
        results.push((additions, multiplications));
    }
    results
}


fn measure_operation_count_lu_binet_strassen(m_algo: MultiplicationAlgorithm, sizes: Vec<usize>) -> Vec<(usize, usize)> {
    let mut results = vec![];
    for &size in sizes.iter() {
        let matrix_a = generate_nxn_matrix(size);

        let mut counter = Counter::new();
        let _result = recursive_lu(&matrix_a,  &mut counter,m_algo);

        let (additions, multiplications, _divisions) = counter.get_count();
        println!("Size: {}, Additions: {}, Multiplications: {}", size, additions, multiplications);
        results.push((additions, multiplications));
    }
    results
}