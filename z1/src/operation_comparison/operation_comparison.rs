use ndarray::{Array2};
use z1::{Counter,generate_nxn_matrix,OperationCounter,binet_multiplication,strassen_multiplication,laderman_multiplication,generate_sizes};
use csv::Writer;


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




// function used during optimization tests to find the best end point for Strassen and Laderman algorithms,
// now must be commented out due to change of functions signature.
// fn landerman_strassen_optimalization_test(){
//     let end_strassen = vec![2,4,8,16,32,64,128,256,512];
//     let end_landerman = vec![3,9,27,81,243];
//     let size_strassen = 1024;
//     let size_landerman = 729;

//     let mut res_strassen = vec![];
//     let mut res_landerman = vec![];

//     for end_point in end_strassen{
//         let mut lowest_duration = std::time::Duration::MAX.as_nanos();

//         for _ in 0..5{
//             let matrix_a = generate_nxn_matrix(size_strassen);
//             let matrix_b = generate_nxn_matrix(size_strassen);
//             let start = std::time::Instant::now();
//             let _res = strassen_multiplication(&matrix_a, &matrix_b, &mut EmptyCounter::new(), end_point);
//             let duration = start.elapsed().as_nanos();
//             if duration < lowest_duration {
//                 lowest_duration = duration;
//             }
//         }
//         println!("Strassen multiplication for size {} took {:?}", end_point, lowest_duration);
//         res_strassen.push((end_point, lowest_duration));
//     }
//     for end_point in end_landerman{
//         let mut lowest_duration = std::time::Duration::MAX.as_nanos();

//         for _ in 0..5{
//             let matrix_a = generate_nxn_matrix(size_landerman);
//             let matrix_b = generate_nxn_matrix(size_landerman);
//             let start = std::time::Instant::now();
//             let _res = laderman_multiplication(&matrix_a, &matrix_b, &mut EmptyCounter::new(), end_point);
//             let duration = start.elapsed().as_nanos();
//             if duration < lowest_duration {
//                 lowest_duration = duration;
//             }
//         }
//         println!("Laderman multiplication for size {} took {:?}", end_point, lowest_duration);
//         res_landerman.push((end_point, lowest_duration));
//     }

//     println!("Strassen results: {:?}", res_strassen);
//     println!("Laderman results: {:?}", res_landerman);

//     let mut wtr = Writer::from_path("z1/src/operation_comparison/strassen_optimization_results.csv").unwrap();

//     for (size, duration) in res_strassen {
//         wtr.write_record(&[size.to_string(), duration.to_string()]).unwrap();
//     }

//     let mut wtr = Writer::from_path("z1/src/operation_comparison/landerman_optimization_results.csv").unwrap();

//     for (size, duration) in res_landerman {
//         wtr.write_record(&[size.to_string(), duration.to_string()]).unwrap();
//     }

//     wtr.flush().unwrap();

// }
