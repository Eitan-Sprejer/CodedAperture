

import numpy as np
from scipy.ndimage import convolve
from utils import SlitScreen, SensorScreen, Decoder
import scipy as sp


def get_mura_decoding_pattern(slit_mask: np.ndarray):
    """
    Calculates the decoding pattern for a MURA mask.

    Args:
        slit_mask (np.ndarray): The MURA mask.

    Returns:
        np.ndarray: The decoding pattern.
    """
    decoding_pattern = np.zeros_like(slit_mask)
    for i in range(slit_mask.shape[0]):
        for j in range(slit_mask.shape[1]):
            cent_i = int(i - (slit_mask.shape[0] - 1) / 2)
            cent_j = int(j - (slit_mask.shape[1] - 1) / 2)
            if cent_i + cent_j == 0:
                decoding_pattern[i, j] = 1
            else:
                if slit_mask[i, j] == 1:
                    decoding_pattern[i, j] = 1
                elif slit_mask[i, j] == 0:
                    decoding_pattern[i, j] = -1
    # Renormalize the decoding pattern for the convolution
    decoding_pattern = decoding_pattern / np.sum(decoding_pattern)
    return decoding_pattern


def get_general_decoding_pattern(slit_mask: np.ndarray):
    """
    Calculates the decoding pattern for a general mask.

    Args:
        slit_mask (np.ndarray): The general mask.

    Returns:
        np.ndarray: The decoding pattern.
    """
    transparency = np.sum(slit_mask) / (slit_mask.shape[0] * slit_mask.shape[1])
    decoding_pattern = np.zeros_like(slit_mask)
    for i in range(slit_mask.shape[0]):
        for j in range(slit_mask.shape[1]):
            if slit_mask[i, j] == 1:
                decoding_pattern[i, j] = 1
            elif slit_mask[i, j] == 0:
                decoding_pattern[i, j] = transparency / (transparency - 1)
    # Renormalize the decoding pattern for the convolution
    decoding_pattern = decoding_pattern / np.sum(decoding_pattern)
    return decoding_pattern


def get_fourier_decoding_pattern(
    slit_mask: np.ndarray, image: np.ndarray, threshold: float
):
    """
    Calculates the decoding pattern for a Fourier mask.

    Args:
        slit_mask (np.ndarray): The Fourier mask.
        image (np.ndarray): The image to be reconstructed.
        threshold (float): The threshold value.

    Returns:
        np.ndarray: The decoding pattern.
    """
    slit_mask = slit_mask / np.sum(slit_mask)
    # Pad the slit_mask with 0s so that it matches the size of the image
    slit_mask = np.pad(
        slit_mask,
        (
            (
                image.shape[0] // 2 - slit_mask.shape[0] // 2,
                image.shape[0] // 2 - slit_mask.shape[0] // 2,
            ),
            (
                image.shape[1] // 2 - slit_mask.shape[1] // 2,
                image.shape[1] // 2 - slit_mask.shape[1] // 2,
            ),
        ),
        "constant",
        constant_values=0,
    )
    slit_mask = np.roll(slit_mask, image.shape[0] // 2, axis=0)
    slit_mask = np.roll(slit_mask, image.shape[1] // 2, axis=1)
    if slit_mask.shape != image.shape:
        slit_mask = slit_mask[: image.shape[0], : image.shape[1]]  # temporary solution

    # Fourier transform the image and the slit
    slit_ft = sp.fft.fft2(slit_mask)
    slit_ft_inv = np.conj(slit_ft) / (np.abs(slit_ft) ** 2 + threshold)
    return slit_ft_inv


def mura_image_reconstruction(sensor: SensorScreen, slit: SlitScreen):
    """
    Reconstructs an image using a MURA mask.

    Args:
        sensor (SensorScreen): The sensor screen.
        slit (SlitScreen): The MURA mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The decoding pattern and the reconstructed image.
    """
    decoding_pattern = get_mura_decoding_pattern(slit.mask)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode="wrap")
    return decoding_pattern, reconstructed_image


def general_image_reconstruction(sensor: SensorScreen, slit: SlitScreen):
    """
    Reconstructs an image using a general mask.

    Args:
        sensor (SensorScreen): The sensor screen.
        slit (SlitScreen): The general mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The decoding pattern and the reconstructed image.
    """
    decoding_pattern = get_general_decoding_pattern(slit.mask)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode="wrap")
    return decoding_pattern, reconstructed_image


def fourier_image_reconstruction(
    sensor: SensorScreen, slit: SlitScreen, threshold: float = 1e-2
):
    """
    Reconstructs an image using a Fourier mask.

    Args:
        sensor (SensorScreen): The sensor screen.
        slit (SlitScreen): The Fourier mask.
        threshold (float, optional): The threshold value. Defaults to 1e-2.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The decoding pattern and the reconstructed image.
    """
    decoding_pattern_ft = get_fourier_decoding_pattern(
        slit.mask, sensor.screen, threshold=threshold
    )
    sensor_ft = sp.fft.fft2(sensor.screen)
    reconstructed_image_ft = sensor_ft * decoding_pattern_ft
    reconstructed_image = sp.fft.ifft2(reconstructed_image_ft)
    decoding_pattern = sp.fft.ifft2(decoding_pattern_ft)
    # Center back the decoding_pattern
    decoding_pattern = np.roll(decoding_pattern, -sensor.screen.shape[0] // 2, axis=0)
    decoding_pattern = np.roll(decoding_pattern, -sensor.screen.shape[1] // 2, axis=1)
    return np.abs(decoding_pattern), np.abs(reconstructed_image)


def decode_image(
    sensor: SensorScreen, slit: SlitScreen, decoder: Decoder, mask_type: str
):
    """
    Decodes an image using a mask.

    Args:
        sensor (SensorScreen): The sensor screen.
        slit (SlitScreen): The mask.
        decoder (Decoder): The decoder.
        mask_type (str): The type of mask.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The decoding pattern and the reconstructed image.
    """
    if decoder.method == "mura":
        assert (
            mask_type == "mura"
        ), "Mura decoding method can only be used with Mura masks"
        decoding_pattern, reconstructed_image = mura_image_reconstruction(sensor, slit)
    elif decoder.method == "general":
        decoding_pattern, reconstructed_image = general_image_reconstruction(
            sensor, slit
        )
    elif decoder.method == "fourier":
        decoding_pattern, reconstructed_image = fourier_image_reconstruction(
            sensor, slit, threshold=decoder.threshold
        )

    # Flip the reconstructed image
    reconstructed_image = np.flip(reconstructed_image, axis=0)
    reconstructed_image = np.flip(reconstructed_image, axis=1)

    return decoding_pattern, reconstructed_image
