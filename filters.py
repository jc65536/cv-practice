import numpy as np

def gaussian_filter(size: int, sigma: float):
    assert size % 2 == 1

    half_size = size // 2
    two_sigma_sq = 2 * sigma ** 2
    factor = 1 / (np.pi * two_sigma_sq)

    # Gaussian function
    g = lambda x, y: factor * np.exp(-(x ** 2 + y ** 2) / two_sigma_sq)

    result = np.zeros((size, size))
    weight_sum = 0 # For normalization

    for i in range(0, size):
        for j in range(0, size):
            weight = g(i - half_size, j - half_size)
            result[i][j] = weight
            weight_sum += weight

    return result / weight_sum / 100

def convolve(img: np.array, kernel: np.array):
    kernel_size, _ = np.shape(kernel)
    pad_size = kernel_size // 2
    img_h, img_w, _ = np.shape(img)
    result = np.zeros((img_h + kernel_size - 1, img_w + kernel_size - 1, 3))

    # Accumulate shifted and scaled images (by linearity)
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            pad_width = ((i, kernel_size - 1 - i),
                         (j, kernel_size - 1 - j),
                         (0, 0))
            result += np.pad(img, pad_width) * kernel[i][j]

    # Crop to original size
    return result[pad_size:-pad_size, pad_size:-pad_size, :]

