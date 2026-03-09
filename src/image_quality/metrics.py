import cv2
import numpy as np

def laplacian_variance(image):
    """
    Blur metric using variance of Laplacian.
    Higher value = sharper image.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lap = cv2.Laplacian(image, cv2.CV_64F)
    return lap.var()

def brightness_mean(image):
    """
    Average brightness of image.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return float(np.mean(image))


def contrast_std(image):
    """
    Contrast estimated as pixel standard deviation.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return float(np.std(image))


def noise_residual(image):
    """
    Estimate noise using high-frequency residual.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(image, (5,5), 0)
    residual = image.astype(np.float32) - blur.astype(np.float32)

    return float(np.std(residual))


def high_frequency_energy(image):

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center = (h//2, w//2)

    radius = min(center) // 2

    high_freq = magnitude.copy()

    high_freq[
        center[0]-radius:center[0]+radius,
        center[1]-radius:center[1]+radius
    ] = 0

    return float(high_freq.sum() / magnitude.sum())


def edge_density(image):

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(image, 100, 200)

    return float(np.sum(edges > 0) / edges.size)