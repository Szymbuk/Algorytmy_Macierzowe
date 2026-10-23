use crate::*;


pub fn matrix_invert_binet(a: &Array2<f64>, counter: &mut impl OperationCounter) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    if n == 1 {
        return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
    }

    // 2. Obsługa nieparzystego wymiaru (dynamiczne powiększenie do parzystego)
    let (a_work, is_odd, _original_n) = make_even(&a);

    let actual_n = a_work.nrows();

    let mid = actual_n / 2;

    // Podział macierzy roboczych na 4 równe bloki
    let a11 = a_work.slice(s![0..mid, 0..mid]);
    let a12 = a_work.slice(s![0..mid, mid..actual_n]);
    let a21 = a_work.slice(s![mid..actual_n, 0..mid]);
    let a22 = a_work.slice(s![mid..actual_n, mid..actual_n]);

    // inverse(a11)
    let a11_inv = matrix_invert_binet(&a11.to_owned(), counter)?;
    // S22 = A22 - A21 * inverse(A11) * A12
    let s22 = &a22 - binet_multiplication(&(binet_multiplication(&(a21.to_owned()),&a11_inv, counter).to_owned()),&(a12.to_owned()), counter); 
    // S22_inv = inverse(S22)
    let s22_inv = matrix_invert_binet(&s22, counter)?;
    // B11 = inverse(A11) * (I + A12 * S22_inv * A21 * inverse(A11))  
    let i = Array2::eye(a11_inv.nrows());
    let a12_s22 = binet_multiplication(&(a12.to_owned()), &s22_inv, counter);
    let a12_s22_a21 = binet_multiplication(&(a12_s22.to_owned()), &(a21.to_owned()), counter);
    let a12_s22_a21_a11_inv = binet_multiplication(&(a12_s22_a21.to_owned()), &(a11_inv.to_owned()), counter);
    let b11 = binet_multiplication(&a11_inv, &(i + a12_s22_a21_a11_inv), counter);

    // B12 = - inverse(A11) * A12 * inverse(S22)
    let b12 = -binet_multiplication(&(binet_multiplication(&(a11_inv.to_owned()), &(a12.to_owned()), counter)), &s22_inv, counter);
    // B21 = - S22_inv * A21 * inverse(A11) 
    let b21 = -binet_multiplication(&(binet_multiplication(&(s22_inv.to_owned()), &(a21.to_owned()), counter)), &a11_inv, counter);
    // B22 = S22_inv
    let b22 = s22_inv;
    // // M6 = (A21 - A11) * (B11 + B12)
    // let m6 = strassen_multiplication(&(a21.to_owned() - &a11), &(b11.to_owned() + &b12),counter);
    // // M7 = (A12 - A22) * (B21 + B22)
    // let m7 = strassen_multiplication(&(a12.to_owned() - &a22), &(b21.to_owned() + &b22),counter);

    // // Składanie bloków wynikowych C
    // let c11 = &m1 + &m4 - &m5 + &m7;
    // let c12 = &m3 + &m5;
    // let c21 = &m2 + &m4;
    // let c22 = &m1 - &m2 + &m3 + &m6;

    counter.increment_additions(a11.nrows() * a11.ncols() * 18);

    let mut a_inv = Array2::zeros((actual_n, actual_n));
    a_inv.slice_mut(s![0..mid, 0..mid]).assign(&b11);
    a_inv.slice_mut(s![0..mid, mid..actual_n]).assign(&b12);
    a_inv.slice_mut(s![mid..actual_n, 0..mid]).assign(&b21);
    a_inv.slice_mut(s![mid..actual_n, mid..actual_n]).assign(&b22);

    // 4. Obcięcie wyniku do oryginalnego rozmiaru, jeśli wymiar wejściowy był nieparzysty
    if is_odd {
        Ok(a_inv.slice(s![0..n, 0..n]).to_owned())
    } else {
        Ok(a_inv)
    }


}


// pub fn matrix_invert_strassen(a: Array2<f64>, counter: &mut impl OperationCounter) -> Result<Array2<f64>, String> {
//     let n = a.nrows();
//     if n != a.ncols() {
//         return Err("Matrix must be square".to_string());
//     }

//     if n == 1 {
//         return Ok(Array2::from_elem((1, 1), 1.0 / a[[0, 0]]));
//     }

//     let mid = n / 2;
//     let a11 = a.slice(s![..mid, ..mid]).to_owned();
//     let a12 = a.slice(s![..mid, mid..]).to_owned();
//     let a21 = a.slice(s![mid.., ..mid]).to_owned();
//     let a22 = a.slice(s![mid.., mid..]).to_owned();

//     let inv_a11 = matrix_invert_strassen(a11, counter)?;
//     let inv_a22 = matrix_invert_strassen(a22, counter)?;

//     let s1 = &a21.dot(&inv_a11);
//     let s2 = &inv_a22.dot(&a12);
//     let s3 = &s1.dot(&s2);

//     let inv_a11_s3 = &inv_a11 - &s3;

//     let mut result = Array2::<f64>::zeros((n, n));
//     result.slice_mut(s![..mid, ..mid]).assign(&inv_a11_s3);
//     result.slice_mut(s![..mid, mid..]).assign(&(-&inv_a11.dot(&a12).dot(&inv_a22)));
//     result.slice_mut(s![mid.., ..mid]).assign(&(-&inv_a22.dot(&a21).dot(&inv_a11)));
//     result.slice_mut(s![mid.., mid..]).assign(&inv_a22);

//     Ok(result)
// }