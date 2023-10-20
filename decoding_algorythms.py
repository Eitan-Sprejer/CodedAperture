import numpy as np
from scipy.ndimage import convolve
from utils import SlitScreen, SensorScreen, Decoder
import scipy as sp

def get_mura_decoding_pattern(slit_mask: np.ndarray):
    decoding_pattern = np.zeros_like(slit_mask)
    for i in range(slit_mask.shape[0]):
        for j in range(slit_mask.shape[1]):
            cent_i = int(i - (slit_mask.shape[0]-1)/2)
            cent_j = int(j - (slit_mask.shape[1]-1)/2)
            if cent_i+cent_j == 0:
                decoding_pattern[i, j] = 1
            else:
                if slit_mask[i, j] == 1:
                    decoding_pattern[i, j] = 1
                elif slit_mask[i, j] == 0:
                    decoding_pattern[i, j] = -1
    # Renormalize the decoding pattern for the convolution
    decoding_pattern /= np.sum(decoding_pattern)
    return decoding_pattern

def get_general_decoding_pattern(slit_mask: np.ndarray):
    transparency = np.sum(slit_mask)/(slit_mask.shape[0]*slit_mask.shape[1])
    decoding_pattern = np.zeros_like(slit_mask)
    for i in range(slit_mask.shape[0]):
        for j in range(slit_mask.shape[1]):
            if slit_mask[i, j] == 1:
                decoding_pattern[i, j] = 1
            elif slit_mask[i, j] == 0:
                decoding_pattern[i, j] = transparency/(transparency-1)
    # Renormalize the decoding pattern for the convolution
    decoding_pattern /= np.sum(decoding_pattern)
    return decoding_pattern

def get_fourier_decoding_pattern(slit_mask: np.ndarray, image: np.ndarray, threshold: float):
    slit_mask /= np.sum(slit_mask)
    # Pad the slit_mask with 0s so that it matches the size of the image
    slit_mask = np.pad(slit_mask, (0, image.shape[0] - slit_mask.shape[0]), mode='constant')
    # Shift the slit_mask so that the center is at the center of the image
    slit_mask = sp.fft.ifftshift(slit_mask)

    # Fourier transform the image and the slit
    slit_ft = sp.fft.fft2(slit_mask)
    slit_ft_inv = np.conj(slit_ft) / (np.abs(slit_ft) ** 2 + threshold)
    return slit_ft_inv

def mura_image_reconstruction(sensor: SensorScreen, slit: SlitScreen):
    decoding_pattern = get_mura_decoding_pattern(slit.mask)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode='wrap')
    return decoding_pattern, reconstructed_image

def general_image_reconstruction(sensor: SensorScreen, slit: SlitScreen):
    decoding_pattern = get_general_decoding_pattern(slit.mask)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode='wrap')
    return decoding_pattern, reconstructed_image

def fourier_image_reconstruction(sensor: SensorScreen, slit: SlitScreen, threshold: float=1e-2):
    decoding_pattern_ft = get_fourier_decoding_pattern(
        slit.mask,
        sensor.screen,
        threshold=threshold
    )
    sensor_ft = sp.fft.fft2(sensor.screen)
    reconstructed_image_ft = sensor_ft * decoding_pattern_ft
    reconstructed_image = sp.fft.ifft2(reconstructed_image_ft)
    decoding_pattern = sp.fft.ifft2(decoding_pattern_ft)
    return np.abs(decoding_pattern), np.abs(reconstructed_image)

def decode_image(sensor: SensorScreen, slit: SlitScreen, decoder: Decoder, mask_type: str):
    if decoder.method == 'mura':
        assert mask_type == 'mura', "Mura decoding method can only be used with Mura masks"
        decoding_pattern, reconstructed_image = mura_image_reconstruction(sensor, slit)
    elif decoder.method == 'general':
        decoding_pattern, reconstructed_image = general_image_reconstruction(sensor, slit)
    elif decoder.method == 'fourier':
        decoding_pattern, reconstructed_image = fourier_image_reconstruction(sensor, slit, threshold=decoder.threshold)

    return decoding_pattern, reconstructed_image
