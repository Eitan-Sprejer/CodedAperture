import numpy as np
from scipy.ndimage import convolve
from utils import SourceScreen, SlitScreen, SensorScreen

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

def decode_image(sensor: SensorScreen, slit: SlitScreen, mask_type: str):
    if mask_type == 'mura':
        decoding_pattern = get_mura_decoding_pattern(slit.mask)
    else:
        decoding_pattern = get_general_decoding_pattern(slit.mask)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode='wrap')
    return decoding_pattern, reconstructed_image
