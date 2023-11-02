

"""
This module provides functions for calculating and visualizing metrics related to 
the reconstruction of an image from a coded aperture. The metrics include mean squared 
error (MSE), peak signal to noise ratio (PSNR), and histogram spread (HS). 
The module also provides functions for renormalizing images, which can be used as 
input to the metric functions. 
Finally, the module includes a function for plotting the intensity profile of the 
screen.

Functions:
- minmax_image_renormalization(image: np.ndarray) -> np.ndarray
- median_zscore_image_renormalization(image: np.ndarray) -> np.ndarray
- get_metrics(source: SourceScreen, decoder: Decoder, renormalization_method: Callable) -> Dict[str, float]
- calculate_mse(source: SourceScreen, decoder: Decoder, renormalization_method: Callable) -> float
- calculate_psnr(source: SourceScreen, decoder: Decoder, renormalization_method: Callable) -> float
- calculate_histogram_spread(source: SourceScreen, decoder: Decoder, renormalization_method: Callable) -> float
- plot_intensity_profile(source: SourceScreen, decoder: Decoder, renormalization_method: Callable, nbins=100) -> None
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from utils import SourceScreen, Decoder


def minmax_image_renormalization(image: np.ndarray):
    """
    Renormalizes the image so that the values are between 0 and 1.
    """
    image = image - np.min(image)
    image = image / np.max(image)
    return image


def median_zscore_image_renormalization(image: np.ndarray):
    """
    Renormalization method sensitive to outliers.
    """
    image = image - np.median(image)
    image = image / (np.percentile(image, 84) - np.percentile(image, 16))
    return image


def get_metrics(
    source: SourceScreen, decoder: Decoder, renormalization_method: Callable
):
    """
    Returns a dictionary with the metrics for the reconstruction.
    """
    metrics = {}
    metrics["mse"] = calculate_mse(
        source, decoder, renormalization_method=renormalization_method
    )
    metrics["psnr"] = calculate_psnr(
        source, decoder, renormalization_method=renormalization_method
    )
    metrics["hs"] = calculate_histogram_spread(
        source, decoder, renormalization_method=renormalization_method
    )
    return metrics


def calculate_mse(
    source: SourceScreen, decoder: Decoder, renormalization_method: Callable
):
    """
    Calculates the mean squared error between the renormalized sensor screen and
    reconstructed image.
    Returns a value between 0 and 1, the lower the better the reconstruction is.
    """
    renormalized_source = renormalization_method(source.screen)
    renormalized_decoded_image = renormalization_method(
        decoder.decoded_image
    )
    mse = np.mean((renormalized_source - renormalized_decoded_image) ** 2)
    return mse


def calculate_psnr(
    source: SourceScreen, decoder: Decoder, renormalization_method: Callable
):
    """
    Calculates the peak signal to noise ratio between the renormalized sensor screen
    and reconstructed image.
    """
    mse = calculate_mse(source, decoder, renormalization_method=renormalization_method)
    psnr = 10 * np.log10(1 / mse)
    return psnr


def calculate_histogram_spread(
    source: SourceScreen, decoder: Decoder, renormalization_method: Callable
):
    """
    Calculates the histogram spread between the renormalized sensor screen and
    reconstructed image.
    The closer to 1, the better the reconstruction is.
    """
    renormalized_source = renormalization_method(source.screen)
    renormalized_decoded_image = renormalization_method(
        decoder.decoded_image
    )
    renormalized_source_hs = np.quantile(renormalized_source, 0.75) - np.quantile(
        renormalized_source, 0.25
    )
    renormalized_decoded_image_hs = np.quantile(
        renormalized_decoded_image, 0.75
    ) - np.quantile(renormalized_decoded_image, 0.25)

    hs_ratio = renormalized_decoded_image_hs / renormalized_source_hs
    return hs_ratio


def plot_intensity_profile(
    source: SourceScreen, decoder: Decoder, renormalization_method: Callable, nbins=100
):
    """
    Plots the intensity profile of the screen.
    """

    renormalized_decoded_image = renormalization_method(
        decoder.decoded_image
    )
    renormalized_source = renormalization_method(source.screen)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.hist(
        renormalized_source.flatten(), bins=nbins, label="Sensor screen", ax=axs[0]
    )
    plt.hist(
        renormalized_decoded_image.flatten(),
        bins=nbins,
        label="Sensor screen",
        ax=axs[1],
    )
    axs[0].set_xlabel("Normalized Intensity")
    axs[1].set_xlabel("Normalized Intensity")
    axs[0].set_ylabel("Frequency")
    axs[1].set_ylabel("Frequency")
    plt.legend()
    plt.show()
