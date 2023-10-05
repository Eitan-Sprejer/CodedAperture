import numpy as np
from scipy.ndimage import convolve
from utils import SourceScreen, SlitScreen, SensorScreen

def get_mura_decoding_pattern(slit: SlitScreen):
    decoding_pattern = np.zeros_like(slit.mask)
    for i in range(slit.mask.shape[0]):
        for j in range(slit.mask.shape[1]):
            cent_i = int(i - (slit.mask.shape[0]-1)/2)
            cent_j = int(j - (slit.mask.shape[1]-1)/2)
            if cent_i+cent_j == 0:
                decoding_pattern[i, j] = 1
            else:
                if slit.mask[i, j] == 1:
                    decoding_pattern[i, j] = 1
                elif slit.mask[i, j] == 0:
                    decoding_pattern[i, j] = -1
    return decoding_pattern

def get_general_decoding_pattern(slit: SlitScreen):
    transparency = np.sum(slit.mask)/(slit.mask.shape[0]*slit.mask.shape[1])
    decoding_pattern = np.zeros_like(slit.mask)
    for i in range(slit.mask.shape[0]):
        for j in range(slit.mask.shape[1]):
            if slit.mask[i, j] == 1:
                decoding_pattern[i, j] = 1
            elif slit.mask[i, j] == 0:
                decoding_pattern[i, j] = transparency/(transparency-1)
    return decoding_pattern

def decode_image(sensor: SensorScreen, slit: SlitScreen, mask_type: str):
    if mask_type == 'mura':
        decoding_pattern = get_mura_decoding_pattern(slit)
    else:
        decoding_pattern = get_general_decoding_pattern(slit)
    reconstructed_image = convolve(sensor.screen, decoding_pattern, mode='wrap')
    return decoding_pattern, reconstructed_image
