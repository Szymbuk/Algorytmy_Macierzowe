use ndarray::Array2;
use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};

pub fn save_matrix_to_csv(matrix: &Array2<f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    // Tworzymy writera bez nagłówków (czyste dane macierzy)
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
    
    // Zapisuje całą strukturę ndarray prosto do pliku
    writer.serialize_array2(matrix)?;
    Ok(())
}

pub fn load_matrix_from_csv(path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    
    // Odczytuje plik i automatycznie parsuje go do Array2<f64>
    let matrix: Array2<f64> = reader.deserialize_array2_dynamic()?;
    Ok(matrix)
}