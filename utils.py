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
import cv2

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


def get_config_name(config_path: str):
    if "/" in config_path:
        config_filename = config_path.split("/")[-1].split(".json")[0]
    elif "\\" in config_path:
        config_filename = config_path.split("\\")[-1].split(".json")[0]
    else:
        config_filename = config_path.split(".json")[0]
    return config_filename


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
    decoder_config = config["decoder"]
    options_config = config["options"]
    # Add the file name to the options
    options_config["config_filename"] = get_config_name(config_path)

    source = SourceScreen(**source_config)
    slit = SlitScreen(**slit_config)
    sensor = SensorScreen(**sensor_config)
    decoder = Decoder(**decoder_config)
    options = Options(**options_config)

    return source, slit, sensor, decoder, options


def split_photons(n_photons: int, n_cores: int) -> list:
    """Splits the number of photons to be simulated between the cores."""
    base_value = n_photons // n_cores
    remainder = n_photons % n_cores
    result = [base_value] * n_cores
    for i in range(remainder):
        result[i] += 1
    return result


def zoom_out_image(image: np.ndarray, zoom_out_factor: float):
    """
    Zooms out an image by a factor of zoom_out_factor, by taking the average of blocks of pixels,
    and then zero padding the edges to take it back to the original size.

    Parameters
    ----------
    image : np.ndarray
        The image to be zoomed out
    zoom_out_factor : int
        The factor by which the image should be zoomed out. Must be an integer greater than 1.

    Returns
    -------
    np.ndarray
        The zoomed out image
    """

    zoom_out_factor = int(zoom_out_factor)

    # Calculate the new dimensions
    new_rows = image.shape[0] // zoom_out_factor
    new_cols = image.shape[1] // zoom_out_factor

    # Create an empty image with the new dimensions
    downsampled_image = np.empty((new_rows, new_cols), dtype=image.dtype)

    # Populate the downsampled image by taking the average of blocks from the original image
    for i in range(new_rows):
        for j in range(new_cols):
            block = image[
                i * zoom_out_factor : (i + 1) * zoom_out_factor,
                j * zoom_out_factor : (j + 1) * zoom_out_factor,
            ]
            downsampled_image[i, j] = np.mean(block)

    # Zero pad the edges to take the downsampled image back to the original size, centering it
    padded_image = np.zeros_like(image)

    # Calculate the starting and ending indices for the padded image
    start_row = (padded_image.shape[0] - downsampled_image.shape[0]) // 2
    end_row = start_row + downsampled_image.shape[0]
    start_col = (padded_image.shape[1] - downsampled_image.shape[1]) // 2
    end_col = start_col + downsampled_image.shape[1]

    # Place the downsampled image in the padded image
    padded_image[start_row:end_row, start_col:end_col] = downsampled_image

    # Plot the downsampled image
    plt.imshow(padded_image)
    return padded_image


def zoom_in_image(image: np.ndarray, zoom_in_factor: float):
    # Calculate the new size while maintaining the same resolution
    new_width = int(image.shape[1] * zoom_in_factor)
    new_height = int(image.shape[0] * zoom_in_factor)

    # Resize the image using OpenCV
    zoomed_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_CUBIC
    )

    # Calculate the cropping parameters with floating-point values
    left = (new_width - image.shape[1]) / 2.0
    top = (new_height - image.shape[0]) / 2.0
    right = left + image.shape[1]
    bottom = top + image.shape[0]

    # Round the cropping parameters to the nearest integers
    left, top, right, bottom = map(int, [left, top, right, bottom])

    # Crop the zoomed image to maintain the original resolution
    zoomed_image = zoomed_image[top:bottom, left:right]

    # Ensure the resulting image is of the same data type as the original
    zoomed_image = zoomed_image.astype(image.dtype)

    return zoomed_image

@dataclass
class Decoder:
    decode_img: bool
    method: str
    fourier_config: dict

    def __post_init__(self) -> None:
        if self.method == "fourier":
            self.threshold = self.fourier_config["threshold"]


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
    mura_config: dict

    def __post_init__(self) -> None:
        self.mask_size = np.array(self.mask_size)
        self.mask_generator = MaskGenerator(
            mask_size=self.mask_size,
            mask_type=self.mask_type,
            mask_width=self.mask_width,
            mura_config=self.mura_config,
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
    config_filename: str
    name: str
    add_noise: bool
    source_to_slit_distance: float
    slit_to_sensor_distance: float
    inter_pixel_distance: float
    theta_bounds: list
    phi_bounds: list
    automatic_angles: bool
    random_seed: int

    def __post_init__(self) -> None:
        self.theta_bounds = np.array(self.theta_bounds)
        self.phi_bounds = np.array(self.phi_bounds)
        # Unit conversion to radians.
        self.theta_bounds = self.theta_bounds * np.pi / 180
        self.phi_bounds = self.phi_bounds * np.pi / 180

        self.source_to_sensor_distance = (
            self.source_to_slit_distance + self.slit_to_sensor_distance
        )

    def set_angle_bounds(self, source_size, slit_size):
        max_phi = max(
            np.arctan(
                (source_size[0] + slit_size[0]) / (2 * self.source_to_slit_distance)
            ),
            np.arctan(
                (source_size[1] + slit_size[1]) / (2 * self.source_to_slit_distance)
            ),
        )
        # Set the maximum phi as the upper bound
        self.phi_bounds[1] = max_phi


def coordinates2positions(
    mask_shape: np.ndarray, options: Options, coordinates: np.ndarray
):
    """
    Converts the coordinates of a pixel to its position in the source or slit plane.
    """

    return (
        coordinates - mask_shape / 2
    ) * options.inter_pixel_distance  # Center the position coordinates.


def positions2coordinates(
    mask_shape: np.ndarray, options: Options, positions: np.ndarray
):
    """
    Converts the positions of a pixel to its coordinates in the source or slit plane.
    """

    return (
        (positions / options.inter_pixel_distance)  # Center the position coordinates.
        + mask_shape / 2
    ).astype(int)


class MaskGenerator:
    def __init__(self, **mask_config):
        self.mask_size = mask_config["mask_size"]
        self.mask_width = mask_config["mask_width"]
        self.mask_type = mask_config["mask_type"]
        if self.mask_type == "mura":
            self.tile = mask_config["mura_config"]["tile"]
            self.rank = mask_config["mura_config"]["rank"]
            self.center = mask_config["mura_config"]["center"]

    def generate_mask(self):
        """Generates a mask with the specified parameters"""
        if self.mask_type == "phi":
            return self.generate_phi_mask()
        if self.mask_type == "pinhole":
            return self.generate_pinhole_mask()
        if self.mask_type == "frame":
            return self.generate_frame_mask()
        if self.mask_type == "mura":
            return self.generate_apertures_mask()
        if "pattern" in self.mask_type:
            return self.load_slit_from_pixelart()
        if "exp" in self.mask_type:
            return self.load_slit_from_png_image()
        if "training_sources" in self.mask_type:
            return np.loadtxt(f"training_sources/{self.mask_type}")
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
        if not np.array_equal(mask.shape, self.mask_size):
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

        # Check if the processed image is already saved
        # if os.path.isfile(path):
        # mask = np.array(Image.open(path))
        # if mask.shape != self.mask_size:
        #    raise ValueError(
        #        f"Mask size {mask.shape} does not match the specified size {self.mask_size}"
        #    )
        # return mask
        # Load the png image as a mask matrix

        try:
            mask = process_image(
                file_name=file_name, target_size=self.mask_size, invert=True
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File {path} not found. Check if it's correct"
            ) from exc
        return mask

    def generate_apertures_mask(self):
        """Generates a mask from the apertures library"""
        mura = ca.mura(rank=self.rank, tile=self.tile, center=self.center)
        return mura.aperture
