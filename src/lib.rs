use std::ops::AddAssign;
use std::sync::{Arc, Mutex};
use std::thread;

use numpy::ndarray::*;
use numpy::*;
use pyo3::prelude::*;

const THREADS: i32 = 16;

fn _convolve(img: ArrayView3<u8>, kernel: ArrayView2<f64>) -> Array3<u8> {
    let img = img.mapv(f64::from);
    let kernel_size = kernel.shape()[0];

    assert!(kernel_size % 2 == 1);

    let kernel_size_1 = kernel_size - 1;
    let pad_size = (kernel_size / 2) as i32;

    let img_shape = img.shape();
    let img_h = img_shape[0];
    let img_w = img_shape[1];

    let zero_buf = || Array3::<f64>::zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3));

    let mut result = zero_buf();

    let result_arc = Arc::new(Mutex::new(&mut result));

    let kernel_iter = Arc::new(Mutex::new(kernel.indexed_iter()));

    thread::scope(|scope| {
        for _ in 0..THREADS {
            let result = result_arc.clone();
            let kernel_iter = kernel_iter.clone();
            let img = img.clone();

            scope.spawn(move || {
                let mut local_sum = zero_buf();

                while let Some(((i, j), k)) = kernel_iter.lock().unwrap().next() {
                    local_sum
                        .slice_mut(s![i..i + img_h, j..j + img_w, ..])
                        .add_assign(&(*k * &img));
                }

                result.lock().unwrap().add_assign(&local_sum);
            });
        }
    });

    // let mut result = Array3::<f64>::zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3));

    // for i in 0..kernel_size {
    //     for j in 0..kernel_size {
    //         result
    //             .slice_mut(s![i..i + img_h, j..j + img_w, ..])
    //             .add_assign(&(kernel[[i, j]] * &img));
    //     }
    // }

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
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    Ok(())
}
