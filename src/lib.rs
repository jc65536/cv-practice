use std::ops::AddAssign;
use std::sync::Arc;
use std::thread;

use numpy::ndarray::*;
use numpy::*;
use pyo3::prelude::*;

fn _convolve(img: ArrayView3<u8>, kernel: ArrayView2<f64>, threads: u32) -> Array3<u8> {
    let img = img.mapv(f64::from);
    let kernel_size = kernel.shape()[0];

    assert!(kernel_size % 2 == 1);

    let kernel_size_1 = kernel_size - 1;
    let pad_size = (kernel_size / 2) as i32;

    let kernel_entries = kernel_size * kernel_size;
    let chunk_size = kernel_entries / threads as usize + 1;

    let img_shape = img.shape();
    let img_h = img_shape[0];
    let img_w = img_shape[1];

    let zero_buf = move || Array3::<f64>::zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3));

    let mut kernel_iter = kernel.indexed_iter();

    let result = if threads <= 1 {
        kernel_iter.fold(zero_buf(), |mut acc, ((i, j), &k)| {
            acc.slice_mut(s![i..i + img_h, j..j + img_w, ..])
                .add_assign(&(k * &img));
            acc
        })
    } else {
        let img = Arc::new(img);

        (0..threads)
            .map(|_| {
                let img = img.clone();

                let kernel_chunk: Vec<_> = (0..chunk_size)
                    .filter_map(|_| match kernel_iter.next() {
                        Some((idx, &k)) => Some((idx, k)),
                        None => None,
                    })
                    .collect();

                thread::spawn(move || {
                    kernel_chunk
                        .into_iter()
                        .fold(zero_buf(), |mut acc, ((i, j), k)| {
                            acc.slice_mut(s![i..i + img_h, j..j + img_w, ..])
                                .add_assign(&(k * &*img));
                            acc
                        })
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .reduce(|acc, frame| acc + &frame)
            .unwrap()
    };

    result
        .slice(s![pad_size..-pad_size, pad_size..-pad_size, ..])
        .mapv(|f| f as u8)
}

#[pyfunction]
fn convolve<'a>(
    py: Python<'a>,
    img: PyReadonlyArray3<u8>,
    kernel: PyReadonlyArray2<f64>,
    threads: u32,
) -> &'a PyArray3<u8> {
    let img = img.as_array();
    let kernel = kernel.as_array();
    _convolve(img, kernel, threads).to_pyarray(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn cvlib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convolve, m)?)?;
    Ok(())
}
