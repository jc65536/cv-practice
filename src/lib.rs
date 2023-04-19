use numpy::*;
use pyo3::prelude::*;

mod libimpl;

#[pyfunction]
fn convolve<'a>(
    py: Python<'a>,
    img: PyReadonlyArray3<u8>,
    kernel: PyReadonlyArray2<f64>,
    threads: u32,
) -> &'a PyArray3<u8> {
    let img = img.as_array();
    let kernel = kernel.as_array();
    libimpl::convolve(img, kernel, threads).to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn cvlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    Ok(())
}
