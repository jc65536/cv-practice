import numpy as np

ident = np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]]).astype(np.float64)

grad_x = np.array([[0, 0, 0],
                   [-1, 1, 0],
                   [0, 0, 0]]).astype(np.float64)

grad_y = np.array([[0, -1, 0],
                   [0, 1, 0],
                   [0, 0, 0]]).astype(np.float64)

grad_xy = np.array([[0, -1, 0],
                    [-1, 2, 0],
                    [0, 0, 0]]).astype(np.float64)


def gaussian(size: int, sigma: float):
    assert size % 2 == 1

    half_size = size // 2
    two_sigma_sq = 2 * sigma ** 2
    factor = 1 / (np.pi * two_sigma_sq)

    # Gaussian function
    def g(x, y): return factor * np.exp(-(x ** 2 + y ** 2) / two_sigma_sq)

    result = np.zeros((size, size))
    weight_sum = 0  # For normalization

    for i in range(0, size):
        for j in range(0, size):
            weight = g(i - half_size, j - half_size)
            result[i, j] = weight
            weight_sum += weight

    return result / weight_sum


def gaussian_sep(size: int, sigma: float):
    assert size % 2 == 1

    half_size = size // 2
    two_sigma_sq = 2 * sigma ** 2
    factor = 1 / (sigma * np.sqrt(2 * np.pi))

    x = np.arange(size) - half_size
    gx = factor * np.exp(-x ** 2 / two_sigma_sq)
    gx /= np.sum(gx)

    return gx


def convolve(img: np.ndarray, kernel: np.ndarray, _):
    img = img.astype(np.float64)
    kernel_size, _ = np.shape(kernel)

    assert kernel_size % 2 == 1

    kernel_size_1 = kernel_size - 1
    pad_size = kernel_size // 2
    img_h, img_w, _ = np.shape(img)

    result = np.zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3))

    # Accumulate shifted and scaled images (by linearity)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            result[i:i + img_h, j:j + img_w, :] += kernel[i, j] * img

    # Crop to original size
    return result[pad_size:-pad_size, pad_size:-pad_size, :].astype(np.uint8)


def convolve_sep(img: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray, _):
    img = img.astype(np.float64)
    kernel_size, = np.shape(kernel_x)

    assert kernel_size % 2 == 1

    kernel_size_1 = kernel_size - 1
    pad_size = kernel_size // 2
    img_h, img_w, _ = np.shape(img)

    result_x = np.zeros((img_h + kernel_size_1, img_w, 3))

    for i in range(0, kernel_size):
        result_x[i:i + img_h, :, :] += kernel_x[i] * img

    result = np.zeros((img_h + kernel_size_1, img_w + kernel_size_1, 3))

    for j in range(0, kernel_size):
        result[:, j:j + img_w, :] += kernel_y[i] * result_x

    # Crop to original size
    return result[pad_size:-pad_size, pad_size:-pad_size, :].astype(np.uint8)
