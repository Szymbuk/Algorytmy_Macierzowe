fn main(){
    let functions = vec![binet_multiplication, strassen_multiplication, laderman_multiplication];
    let function_names = vec!["Binet", "Strassen", "Laderman"];
    for (func, name) in functions.iter().zip(function_names.iter()) {
        println!("Testing function: {}", name);
        let sizes = generate_sizes(1024 as usize);
        let results = measure_operation_counts(*func, sizes.clone());
        println!();

        let mut wtr = Writer::from_path(format!("z1/src/operation_comparison/{name}_operation_count.csv")).unwrap();
        wtr.write_record(&["Size", "Additions", "Multiplications"]).unwrap();
        for (size, (add, mult)) in sizes.iter().zip(results.iter()) {
            wtr.write_record(&[size.to_string(), add.to_string(), mult.to_string()]).unwrap();
        }

        wtr.flush().unwrap();
    } 
    //landerman_strassen_optimalization_test();
}






fn measure_operation_counts(func: fn(&Array2<f64>, &Array2<f64>, &mut Counter) -> Array2<f64>,sizes: Vec<usize>) -> Vec<(usize, usize)> {
    let mut results = vec![];
    for &size in sizes.iter() {
        let matrix_a = generate_nxn_matrix(size);
        let matrix_b = generate_nxn_matrix(size);

        let mut counter = Counter::new();
        let _result = func(&matrix_a, &matrix_b, &mut counter);

        let (additions, multiplications, _divisions) = counter.get_count();
        println!("Size: {}, Additions: {}, Multiplications: {}", size, additions, multiplications);
        results.push((additions, multiplications));
    }
    results
}