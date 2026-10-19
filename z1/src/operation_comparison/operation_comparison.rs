use ndarray::{Array2, range};
use z1::{EmptyCounter,Counter,generate_nxn_matrix,OperationCounter,binet_multiplication,strassen_multiplication,laderman_multiplication};
use std::error::Error;
use csv::Writer;


fn main(){
    let functions = vec![binet_multiplication, strassen_multiplication, laderman_multiplication];
    let function_names = vec!["Binet", "Strassen", "Laderman"];
    for (func, name) in functions.iter().zip(function_names.iter()) {
        println!("Testing function: {}", name);
        let sizes = generate_sizes_n(1024 as usize);
        let mut results = measure_operation_counts(*func, sizes);
        println!();

        let mut wtr = Writer::from_path(format!("z1/src/operation_comparison/{name}_operation_count.csv")).unwrap();

        for (add, mult) in results {
            wtr.write_record(&[add.to_string(), mult.to_string()]).unwrap();
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


pub fn generate_sizes_n(max_n: usize) -> Vec<usize> {
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
