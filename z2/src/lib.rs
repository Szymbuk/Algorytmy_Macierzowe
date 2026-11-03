pub mod invert_binet_strassen;
pub mod factorizationLU_binet_strassen;
pub mod invert_laderman;
pub use crate::invert_binet_strassen::{recursive_matrix_inversion};
pub use crate::factorizationLU_binet_strassen::{recursive_lu};
//use::z1::{generate_nxn_matrix,generate_seeded_matrix,generate_sizes, OperationCounter,Counter,EmptyCounter,binet_multiplication, strassen_multiplication,laderman_multiplication,dot_product,make_even};
pub use z1::*;
pub use ndarray_linalg::Inverse;
pub use ndarray_linalg::Factorize;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum MatrixType {
    Normal, // Gęsta macierz ogólna (pełny Schur)
    Lower,  // Macierz dolnotrójkątna
    Upper,  // Macierz górnotrójkątna
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum MultiplicationAlgorithm {
    Binet,
    Strassen,
    Laderman,
}

pub struct MatrixMultiplier<'a, C: OperationCounter> {
    counter: &'a mut C,
    algo: MultiplicationAlgorithm,
}

impl<'a, C: OperationCounter> MatrixMultiplier<'a, C> {
    pub fn new(counter: &'a mut C, algo: MultiplicationAlgorithm) -> Self {
        Self { counter, algo }
    }

    /// Jednolity interfejs do mnożenia – ukrywa wybór algorytmu i przekazywanie licznika
    pub fn multiply(&mut self, a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
        match self.algo {
            MultiplicationAlgorithm::Binet => binet_multiplication(a, b, self.counter),
            MultiplicationAlgorithm::Strassen => strassen_multiplication(a, b, self.counter),
            MultiplicationAlgorithm::Laderman => laderman_multiplication(a, b, self.counter),
        }
    }
}

pub fn make_divisible_by_3(a: &Array2<f64>) -> (Array2<f64>, bool, usize) {
    let n = a.nrows();
    let rem = n % 3;
    if rem == 0 {
        return (a.clone(), false, n);
    }
    let target_n = n + (3 - rem);
    let mut padded = Array2::eye(target_n);
    padded.slice_mut(s![0..n, 0..n]).assign(a);

    (padded, true, n)
}