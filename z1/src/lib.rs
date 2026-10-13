use ndarray as nd;



fn main1(){
    let a = nd::Array2::<f64>::zeros((3, 3));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_main1() {
        main1();
    }
}