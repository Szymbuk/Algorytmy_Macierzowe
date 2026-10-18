use ndarray::Array2;
use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder, WriterBuilder};
use ndarray_csv::{Array2Reader, Array2Writer};

pub fn zapisz_macierz_do_csv(macierz: &Array2<f64>, sciezka: &str) -> Result<(), Box<dyn Error>> {
    let plik = File::create(sciezka)?;
    // Tworzymy writera bez nagłówków (czyste dane macierzy)
    let mut writer = WriterBuilder::new().has_headers(false).from_writer(plik);
    
    // Zapisuje całą strukturę ndarray prosto do pliku
    writer.serialize_array2(macierz)?;
    Ok(())
}

pub fn wczytaj_macierz_z_csv(sciezka: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let plik = File::open(sciezka)?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(plik);
    
    // Odczytuje plik i automatycznie parsuje go do Array2<f64>
    let macierz: Array2<f64> = reader.deserialize_array2_dynamic()?;
    Ok(macierz)
}