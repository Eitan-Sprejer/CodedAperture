import json
import numpy as np
from tqdm import tqdm
import os
import shutil
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import timeit
import codedapertures as ca
from scipy import ndimage
from scipy.signal import convolve2d
from typing import Union
from image_preprocessing import process_image
from PIL import Image

from decoding_algorythms import mura_decoding_algorythm

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


def get_config_from_path(path) -> dict:
    """Loads a json file from a path and returns a dictionary with the config"""
    with open(path, "r") as file:
        config = json.load(file)
    return config

def get_objects_from_config(config_path):
    """Returns the objects needed for the simulation from a config file."""

    config = get_config_from_path(config_path)
    source_config = config["source"]
    slit_config = config["slit"]
    sensor_type = config["sensor"]["type"]
    sensor_config = {
        "mask_size": config["sensor"]["mask_size"],
        "exposure_time": config["sensor"]["exposure_time"],
        **config["sensor"][sensor_type],
    }
    options_config = config["options"]

    source = SourceScreen(**source_config)
    slit = SlitScreen(**slit_config)
    sensor = SensorScreen(**sensor_config)
    options = Options(**options_config)

    return source, slit, sensor, options

@dataclass
class SourceScreen:
    mask_size: list
    mask_type: str
    mask_width: int
    photons_per_pixel: int

    def __post_init__(self) -> None:

        self.mask_size = np.array(self.mask_size)
        self.mask_generator = MaskGenerator(
            mask_size=self.mask_size,
            mask_type=self.mask_type,
            mask_width=self.mask_width,
        )
        self.mask = self.mask_generator.generate_mask()
        self.screen = self.mask * self.photons_per_pixel


@dataclass
class SlitScreen:
    mask_size: list
    mask_type: str
    mask_width: int

    def __post_init__(self) -> None:
        
        self.mask_size = np.array(self.mask_size)
        self.mask_generator = MaskGenerator(
            mask_size=self.mask_size,
            mask_type=self.mask_type,
            mask_width=self.mask_width,
        )
        self.mask = self.mask_generator.generate_mask()


@dataclass
class SensorScreen:
    mask_size: list
    readout_noise: float
    dark_current: float
    exposure_time: float

    def __post_init__(self) -> None:
        
        self.mask_size = np.array(self.mask_size)
        self.screen = np.zeros(self.mask_size)
        self.noise_matrix = np.zeros(self.mask_size)
        self.dark_current_noise = self.dark_current * self.exposure_time


@dataclass
class Options:
    name: str
    add_noise: bool
    source_to_slit_distance: float
    slit_to_sensor_distance: float
    inter_pixel_distance: float
    theta_bounds: list
    phi_bounds: list
    random_seed: int

    def __post_init__(self) -> None:

        self.theta_bounds = np.array(self.theta_bounds)
        self.phi_bounds = np.array(self.phi_bounds)
        self.theta_bounds = self.theta_bounds * np.pi / 180
        self.phi_bounds = self.phi_bounds * np.pi / 180

        self.source_to_sensor_distance = (
            self.source_to_slit_distance + self.slit_to_sensor_distance
        )

def coordinates2positions(
    mask_shape: np.ndarray,
    options: Options,
    coordinates: np.ndarray
):
    """
    Converts the coordinates of a pixel to its position in the source or slit plane.
    """

    return (
        (coordinates - mask_shape / 2) # Center the position coordinates.
        * options.inter_pixel_distance
    )

def positions2coordinates(
    mask_shape: np.ndarray,
    options: Options,
    positions: np.ndarray
):
    """
    Converts the positions of a pixel to its coordinates in the source or slit plane.
    """

    return (
        (positions / options.inter_pixel_distance) # Center the position coordinates.
        + mask_shape / 2
    ).astype(int)

class MaskGenerator:
    def __init__(self, **mask_config):
        self.mask_size = mask_config["mask_size"]
        self.mask_width = mask_config["mask_width"]
        self.mask_type = mask_config["mask_type"]

    def generate_mask(self):
        """Generates a mask with the specified parameters"""
        if self.mask_type == "phi":
            return self.generate_phi_mask()
        elif self.mask_type == "pinhole":
            return self.generate_pinhole_mask()
        elif self.mask_type == "frame":
            return self.generate_frame_mask()
        elif self.mask_type == "mura":
            return self.generate_apertures_mask()
        elif "pattern" in self.mask_type:
            return self.load_slit_from_pixelart()
        elif "exp" in self.mask_type:
            return self.load_slit_from_png_image()
        else:
            raise ValueError("Invalid mask type")

    def generate_phi_mask(self):
        """Generates a mask with a phi shape"""
        height, length = self.mask_size
        width = self.mask_width
        mask = np.zeros((height, length))

        mask[
            int(height * 0.1) : int(height * 0.9),
            int((length - width) / 2) : int((length + width) / 2),
        ] = 1
        mask[
            int(height * 0.2) : int(height * 0.4),
            int((length / 3) - (width / 2)) : int((length / 3) + (width / 2)),
        ] = 1
        mask[
            int(height * 0.2) : int(height * 0.4),
            int((2 * length / 3) - (width / 2)) : int((2 * length / 3) + (width / 2)),
        ] = 1
        mask[
            int((height / 3) - (width / 2)) : int((height / 3) - (width / 2)),
            int(length * 0.3) : int(length * 0.7),
        ] = 1
        return mask

    def generate_pinhole_mask(self):
        """Generates a mask with a pinhole shape"""
        height, length = self.mask_size
        mask = np.zeros((height, length))
        hole_radius = int(self.mask_width / 2)
        # Define the lambda shape
        mask[
            int(height * 0.5) - hole_radius : int(height * 0.5) + hole_radius,
            int(length * 0.5) - hole_radius : int(length * 0.5) + hole_radius,
        ] = 1
        return mask

    def generate_frame_mask(self):
        """Generates a mask with a square frame shape"""
        height, length = self.mask_size
        mask = np.zeros((height, length))
        slit_outer_radius = int((np.min([height, length]) + self.mask_width) / 2)
        mask[
            int((height - slit_outer_radius) / 2) : int(
                (height + slit_outer_radius) / 2
            ),
            int((length - slit_outer_radius) / 2) : int(
                (length + slit_outer_radius) / 2
            ),
        ] = 1
        slit_inner_radius = int((np.min([height, length]) - self.mask_width) / 2)

        mask[
            int((height - slit_inner_radius) / 2) : int(
                (height + slit_inner_radius) / 2
            ),
            int((length - slit_inner_radius) / 2) : int(
                (length + slit_inner_radius) / 2
            ),
        ] = 0
        return mask

    def load_slit_from_pixelart(self):
        """
        Generates a mask from a pixel art image.
        The image is first processed to be a binary mask.
        """
        path = f"patterns/{self.mask_type}.png"
        # Load the png image as a mask matrix
        try:
            image = plt.imread(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File {path} not found. Check if it's correct"
            ) from exc
        
        mask = image[:, :, 3]
        # Check that the size is the correct one
        if mask.shape != self.mask_size:
            raise ValueError(
                f"Mask size {mask.shape} does not match the specified size {self.mask_size}"
            )
        return mask

    def load_slit_from_png_image(self):
        """
        Generates a mask from a png image.
        The image is first processed to be a binary mask.
        """
        file_name = self.mask_type
        path = f"exp_pics/{self.mask_type}.png"
        # Load the png image as a mask matrix
        try:
            mask = process_image(file_name= file_name, target_size=self.mask_size, invert=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File {path} not found. Check if it's correct"
            ) from exc
        return mask

    def generate_apertures_mask(self):
        """Generates a mask from the apertures library"""
        mura = ca.mura(rank=4, tile=None, center=True)
        return mura.aperture
