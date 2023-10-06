import os
import shutil
import json
from decoding_algorythms import decode_image
from utils import get_objects_from_config, positions2coordinates, coordinates2positions, Options, SourceScreen, SlitScreen, SensorScreen
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d, correlate
import pickle
import sys
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["legend.fontsize"] = "medium"
plt.rcParams["axes.labelsize"] = "large"


class CodApSimulator:
    def __init__(
        self,
        options: Options,
        source: SourceScreen,
        slit: SlitScreen,
        sensor: SensorScreen,
    ):

        # Initializing attributes.
        self.options = options
        self.source = source
        self.slit = slit
        self.sensor = sensor

        # Post processing attributes
        if options.automatic_angles:
            options.set_angle_bounds(self.source.mask_size, self.slit.mask_size)

        # Initializing results matrices
        self.decoding_pattern = np.zeros_like(self.slit.mask)
        self.decoded_image = np.zeros_like(self.source.mask)

        self.saving_dir = self.get_path_to_save()
        self.rng = np.random.default_rng(seed=options.random_seed)

        # Check whether the saving directory is empty, not to override previous results.
        if os.listdir(self.saving_dir):
            print("The chosen directory is not empty:")
            print(self.saving_dir)
            ans = input(
                "Do you want to continue, override the previous results? (y/n)"
            )
            if ans == "n":
                print("Exiting...")
                sys.exit()
            elif ans == "y":
                print("Continuing...")
            else:
                raise ValueError("Invalid answer")

    def get_path_to_save(self):
        """Returns the path to save the results"""

        os.makedirs("results", exist_ok=True)
        saving_dir = os.path.join("results", self.options.name)
        os.makedirs(saving_dir, exist_ok=True)
        return saving_dir

    def save_results(self, config_path: str):
        """Saves the results of the simulation"""
        np.save(os.path.join(self.saving_dir, "image.npy"), self.sensor.screen)
        # Copy the config file to the results folder
        shutil.copy(config_path, self.saving_dir)

    def add_noise(self):
        """Adds noise to the image"""
        self.sensor.noise_matrix += self.rng.poisson(
            lam=self.sensor.dark_current_noise, size=self.sensor.screen.shape
        )
        self.sensor.noise_matrix += self.rng.normal(
            loc=0, scale=self.sensor.readout_noise, size=self.sensor.screen.shape
        )

        self.sensor.screen += self.sensor.noise_matrix

    def decode_image(self):
        """Decodes the final image generated on the sensor screen."""
        self.decoding_pattern, self.decoded_image = decode_image(self.sensor, self.slit, self.slit.mask_type)

    def make_image(self):
        """Simulates the propagation of photons through the slit"""

        # Find the coordinates of non-zero elements in the source mask
        i_coords, j_coords = np.where(self.source.mask == 1)

        for _ in tqdm(range(self.source.photons_per_pixel)):
            # Sample the angles for the photon
            theta, phi = self.sample_angles()

            photon_theta = theta[i_coords, j_coords]
            photon_phi = phi[i_coords, j_coords]

            # Check if the photons pass through the slit
            passes_through = self.passes_through_slit(
                np.vstack((i_coords, j_coords, np.zeros_like(i_coords))).T,
                np.vstack((photon_theta, photon_phi)).T,
            )

            # Compute the sensor positions for photons that pass through the slit
            valid_photons = passes_through.nonzero()[0]
            landing_positions = self.compute_landing_pixels(
                np.vstack(
                    (
                        i_coords[valid_photons],
                        j_coords[valid_photons],
                        np.zeros_like(valid_photons),
                    )
                ).T,
                np.vstack((photon_theta[valid_photons], photon_phi[valid_photons])).T,
                self.options.source_to_sensor_distance,
            )

            # Increment the sensor screen for valid landing positions
            for sensor_position in landing_positions:
                try:
                    self.sensor.screen[sensor_position[0], sensor_position[1]] += 1
                except IndexError:
                    pass

    def passes_through_slit(self, coordinates, angles):
        """
        Returns a boolean array indicating whether each photon passes through the slit.

        Parameters:
        coordinates: np.array
            The coordinates of the pixels in the source mask for all photons.
        angles: np.array
            The angles of the director vectors for all photons.

        Returns:
        np.array
            A boolean array where True indicates a photon passes through the slit.
        """

        # Compute the intersection pixel coordinates for all photons
        intersection_pixel_coordinates = self.compute_landing_pixels(
            coordinates, angles, self.options.source_to_slit_distance
        )

        # Check if the intersection coordinates are within the slit dimensions for all photons
        within_slit_bounds = (
            (intersection_pixel_coordinates[:, 0] >= 0)
            & (intersection_pixel_coordinates[:, 0] < self.slit.mask.shape[0])
            & (intersection_pixel_coordinates[:, 1] >= 0)
            & (intersection_pixel_coordinates[:, 1] < self.slit.mask.shape[1])
        )

        # Create a boolean array of the same length as positions, initialized with False
        passes_through = np.zeros(coordinates.shape[0], dtype=bool)

        # Set True for the photons that are within slit bounds and the corresponding
        # slit.mask value is 1
        valid_indices = np.where(within_slit_bounds)[0]
        valid_coords = intersection_pixel_coordinates[valid_indices]
        passes_through[valid_indices] = (
            self.slit.mask[valid_coords[:, 0], valid_coords[:, 1]] == 1
        )

        return passes_through

    def compute_landing_pixels(self, coordinates, angles, z_distance):
        """
        Computes the pixels in the sensor screen where the photons land.

        Parameters
        ----------
        positions: np.array
            The positions of the pixels in the source mask.
        angles: np.array
            The angles of the director vectors for the photons.
        z_distance: float
            The distance from the source mask to the sensor screen for all photons.
        """

        theta, phi = angles[:, 0], angles[:, 1]

        # Convert the coordinates on the source to the x-y positions
        positions = coordinates2positions(
            mask_shape=np.array(self.source.mask_size),
            options=self.options,
            coordinates=coordinates[:, :2],
        )

        # Create the origin pixel vectors for all pixels.
        origin_vectors = np.column_stack(
            (
                positions[:, 0],
                positions[:, 1],
                coordinates[:, 2],
            )
        )
        # Create the unit direction pixel vectors for all photons.
        unit_direction_vectors = np.column_stack(
            (np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))
        )
        # Compute the intersections of the pixel directions with the slit screen for all photons.
        intersection_vectors = origin_vectors + unit_direction_vectors * (
            z_distance / unit_direction_vectors[:, 2][:, np.newaxis]
        )
        # Compute the pixel coordinates of the intersections for all photons.
        if z_distance == self.options.source_to_sensor_distance:
            intersection_pixel_coordinates = positions2coordinates(
                mask_shape=self.sensor.mask_size,
                options=self.options,
                positions=intersection_vectors[:, :2],
            )
        elif z_distance == self.options.source_to_slit_distance:
            intersection_pixel_coordinates = positions2coordinates(
                mask_shape=self.slit.mask_size,
                options=self.options,
                positions=intersection_vectors[:, :2],
            )
        else:
            raise ValueError(
                "z_distance must be either source_to_sensor_distance or source_to_slit_distance"
            )
        return intersection_pixel_coordinates

    def sample_angles(self):
        """Samples the angles of the photons from a uniform distribution"""
        theta = self.rng.uniform(*self.options.theta_bounds, self.source.mask.shape)
        phi = self.rng.uniform(*self.options.phi_bounds, self.source.mask.shape)
        return theta, phi


def play_simulation(simulator: CodApSimulator, config_path: str):
    """Plays the simulation"""
    print("Simulating the propagation of photons through the slit...")

    simulator.make_image()
    if simulator.options.add_noise:
        print("Adding noise to the image...")
        simulator.add_noise()
    simulator.decode_image()
    print("Done!")
    print("Saving results...")
    simulator.save_results(config_path)

def plot_results(simulator: CodApSimulator):
    """
    Plots the results of the simulation.

    Parameters
    ----------
    simulator: CodApSimulator
        The simulator object.

    Saves
    -----
    results.png: A 2x2 grid with plots of the source, slit, and sensor screen,
    and the decoded image, if any.

    noise_matrix.png: A plot of the added noise.

    decoding_pattern.png: A plot of the decoding pattern.

    charge_histogram.png: A histogram of the charge values in the sensor screen.
    """
    # Plot the results
    fig = plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    vmin, vmax = 0, float(np.max(simulator.source.screen))
    im = plt.imshow(simulator.source.screen, vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.05, 0.54, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.title("Source Photons")
    plt.subplot(2, 2, 2)
    plt.imshow(simulator.slit.mask, cmap="binary_r")
    plt.title("Slit screen")
    plt.subplot(2, 2, 3)
    vmin, vmax = 0, np.max(simulator.sensor.screen)
    # vmin, vmax = 0, float(np.percentile(simulator.sensor.screen, 99.9))
    im = plt.imshow(simulator.sensor.screen, vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.05, 0.12, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.title("Detected Photons")
    plt.subplot(2, 2, 4)
    vmin, vmax = 0, np.max(simulator.decoded_image)
    # vmin, vmax = 0, float(np.percentile(simulator.decoded_image, 99.9))
    im = plt.imshow(simulator.decoded_image, vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.title("Decoded Image")
    # Save figure
    plt.savefig(os.path.join(simulator.saving_dir, "results.png"))

    # plot the noise matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(simulator.sensor.noise_matrix)
    plt.colorbar()
    plt.savefig(os.path.join(simulator.saving_dir, "noise_matrix.png"))

    plt.figure(figsize=(10, 10))
    plt.hist(
        simulator.sensor.screen.flatten(),
        bins=np.arange(0, np.round(np.max(simulator.sensor.screen)), 0.1)
    )
    plt.savefig(os.path.join(simulator.saving_dir, "charge_histogram.png"))

    plt.figure(figsize=(10, 10))
    plt.imshow(simulator.decoding_pattern)
    plt.savefig(os.path.join(simulator.saving_dir, "decoding_pattern.png"))


def main():
    """
    Main function of the script. Parses the arguments, runs the simulation, and
    saves the results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    config_path = args.config
    source, slit, sensor, options = get_objects_from_config(config_path=config_path)
    simulator = CodApSimulator(source=source, slit=slit, sensor=sensor, options=options)

    play_simulation(simulator=simulator, config_path=config_path)
    plot_results(simulator)

    # Save simulator object as a pickle file.
    with open(os.path.join(simulator.saving_dir, "simulator.pkl"), "wb") as f:
        pickle.dump(simulator, f)

if __name__ == "__main__":
    main()
