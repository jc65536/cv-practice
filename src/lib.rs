use numpy::ndarray::{s, Array3, ArrayView2, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray2, PyReadonlyArray3, ToPyArray};
use pyo3::prelude::*;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

fn _convolve(img: ArrayView3<u8>, kernel: ArrayView2<f64>) -> Array3<u8> {
    let img = img.mapv(f64::from);
    let kernel_size = kernel.shape()[0];

    assert!(kernel_size % 2 == 1);

    let kernel_size_1 = kernel_size - 1;
    let pad_size = (kernel_size / 2) as i32;
    let img_shape = img.shape();
    let img_h = img_shape[0];
    let img_w = img_shape[1];

    let padded_zeros = || Array3::<f64>::zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3));

    let mut result = padded_zeros();

    for i in 0..kernel_size {
        for j in 0..kernel_size {
            let scale_fac = *kernel.get((i, j)).unwrap();

            let mut shifted_img = padded_zeros();

            shifted_img
                .slice_mut(s![i..(i + img_h), j..(j + img_w), ..])
                .assign(&img);

            shifted_img *= scale_fac;

            result += &shifted_img;
        }
    }

    result
        .slice(s![pad_size..-pad_size, pad_size..-pad_size, ..])
        .mapv(|f| f as u8)
}

#[pyfunction]
fn convolve<'a>(
    py: Python<'a>,
    img: PyReadonlyArray3<u8>,
    kernel: PyReadonlyArray2<f64>,
) -> &'a PyArray3<u8> {
    let img = img.as_array();
    let kernel = kernel.as_array();
    _convolve(img, kernel).to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn cvlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    Ok(())
}
