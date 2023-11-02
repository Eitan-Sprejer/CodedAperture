"""
This script runs the simulation of the experiment. It takes a config file as an
argument, which contains the parameters of the simulation. The script creates
the objects needed for the simulation, runs the simulation, and saves the
results.

The config file must be a json file with the following structure:

{
    "source": {
        "mask_size": [int, int],
        "mask_type": str,
        "mask_width": int,
        "photons_per_pixel": int
    },
    "slit": {
        "mask_size": [int, int],
        "mask_type": str,
        "mask_width": int
    },
    "sensor": {
        "type": str,
        "mask_size": [int, int],
        "exposure_time": float,
        "readout_noise": float,
        "dark_current_noise": float,
        "quantum_efficiency": float
    },
    "options": {
        "name": str,
        "source_to_slit_distance": float,
        "source_to_sensor_distance": float,
        "theta_bounds": [float, float],
        "phi_bounds": [float, float],
        "automatic_angles": bool,
        "add_noise": bool,
        "random_seed": int
    }
}

The source, slit, sensor and Options objects are defined in utils.py. 
The simulation is run using the CodApSimulator class defined in this script.

The script saves the results in the results folder. This folder is
created inside the 'results' folder, with the name specified in the config file.
The results folder contains the following files:

image.npy: The image generated on the sensor screen.

config.json: The config file.

results.png: A 2x2 grid with plots of the source, slit, and sensor screen,
and the decoded image, if any.

noise_matrix.png: A plot of the added noise.

decoding_pattern.png: A plot of the decoding pattern.

charge_histogram.png: A histogram of the charge values in the sensor screen.

simulator.pkl: A pickle file containing the simulator object.


Usage
-----
python experiment.py --config path/to/config.json

or, if parallelization is required:
python experiment.py --config path/to/config.json --parallelize
"""

import os
import shutil
import json
from decoding_algorythms import decode_image
from utils import split_photons, get_objects_from_config, positions2coordinates, coordinates2positions, zoom_in_image, Options, SourceScreen, SlitScreen, SensorScreen, Decoder
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d, correlate
import pickle
import sys
import multiprocessing

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
        decoder: Decoder
    ):

        # Initializing attributes.
        self.options = options
        self.source = source
        self.slit = slit
        self.sensor = sensor
        self.decoder = decoder

        # Post processing attributes
        if self.options.automatic_angles:
            self.options.set_angle_bounds(self.source.mask_size, self.slit.mask_size)

        # Initializing results matrices
        self.decoder.decoding_pattern = np.zeros_like(self.slit.mask)
        self.decoder.decoded_image = np.zeros_like(self.source.mask)

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
        saving_dir = os.path.join("results", self.options.config_filename, self.options.name)
        os.makedirs(saving_dir, exist_ok=True)
        return saving_dir

    def save_results(self, config_path: str):
        """Saves the results of the simulation"""
        # Copy the config file to the results folder
        shutil.copy(config_path, self.saving_dir)

        # Save simulator object as a pickle file.
        with open(os.path.join(self.saving_dir, "simulator.pkl"), "wb") as file:
            pickle.dump(self, file)

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
        self.decoder.decoding_pattern, self.decoder.decoded_image = decode_image(
            self.sensor, self.slit, self.decoder, self.slit.mask_type
        )

    def make_image(self, parallelize: bool=False):
        """Simulates the propagation of photons through the slit"""

        if parallelize:
            num_photons_per_pixel = self.source.photons_per_pixel

            num_processes = multiprocessing.cpu_count()  # Get the number of CPU cores

            # Calculate the number of photons to simulate per process
            photons_per_process = split_photons(num_photons_per_pixel, num_processes)

            # Create a pool of worker processes
            pool = multiprocessing.Pool(processes=num_processes)

            # Prepare the arguments for the simulate_photons function
            args_list = [(photons, pbar_pos) for pbar_pos, photons in enumerate(photons_per_process)]

            # Execute the simulation in parallel
            results = pool.map(self.simulate_photons, args_list)
            for result in results:
                self.sensor.screen += result

            # Close the pool of worker processes
            pool.close()
            pool.join()
        else:
            self.simulate_photons((self.source.photons_per_pixel, 0))


    def simulate_photons(self, args: tuple):
        num_photons, pbar_pos = args

        for i in tqdm(range(num_photons), desc=f"Process {os.getpid()}", position=pbar_pos):
            # Sample a matrix of uniform random numbers between 0 and 1 of the same shape as the source mask
            random_matrix = self.rng.uniform(size=self.source.mask.shape)
            # Do bernuli experiment to see it the photons are emited for each pixel
            emitted_photons = (random_matrix < self.source.mask).astype(int)
            
            # Find the coordinates of non-zero elements in the source mask
            i_coords, j_coords = np.where(emitted_photons == 1)
            
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
            for sensor_position, valid_photon in zip(landing_positions, valid_photons):
                try:
                    self.sensor.screen[sensor_position[0], sensor_position[1]] += self.source.mask[i_coords[valid_photon], j_coords[valid_photon]]
                except IndexError:
                    pass
        return self.sensor.screen


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
        theta = np.random.uniform(*self.options.theta_bounds, self.source.mask.shape)
        phi = np.random.uniform(*self.options.phi_bounds, self.source.mask.shape)
        return theta, phi


def play_simulation(simulator: CodApSimulator, config_path: str, parallelize: bool=False):
    """Plays the simulation"""
    print("Simulating the propagation of photons through the slit...")

    simulator.make_image(parallelize=parallelize)
    if simulator.options.add_noise:
        print("Adding noise to the image...")
        simulator.add_noise()
    if simulator.decoder.decode_img:
        print("Decoding Image...")
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
    # Zoom-in the sensor screen and reconstruction
    zoom_factor = simulator.options.source_to_sensor_distance / simulator.options.slit_to_sensor_distance
    simulation_zoom_factor = zoom_factor/(np.max(simulator.source.screen.shape)/np.max(simulator.sensor.screen.shape))
    zoomed_in_decoded_image = zoom_in_image(simulator.decoder.decoded_image, simulation_zoom_factor)

    # Plot the results
    fig = plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    vmin, vmax = 0, float(np.max(simulator.source.screen))
    plt.title("Source Photons")
    im = plt.imshow(simulator.source.screen, vmin=vmin, vmax=vmax, cmap = "viridis")
    cbar_ax = fig.add_axes([0.05, 0.54, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplot(2, 2, 2)
    plt.title("Slit screen")
    plt.imshow(simulator.slit.mask, cmap="binary_r")
    plt.subplot(2, 2, 3)
    vmin, vmax = 0, np.max(simulator.sensor.screen)
    plt.title("Detected Photons")
    im = plt.imshow(simulator.sensor.screen, vmin=vmin, vmax=vmax, cmap = "viridis")
    cbar_ax = fig.add_axes([0.05, 0.12, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    plt.subplot(2, 2, 4)
    vmin, vmax = 0, np.max(zoomed_in_decoded_image)
    plt.title("Zoomed-In Reconstructed Image")
    im = plt.imshow(zoomed_in_decoded_image, vmin=vmin, vmax=vmax, cmap = "viridis")
    cbar_ax = fig.add_axes([0.95, 0.12, 0.01, 0.3])
    fig.colorbar(im, cax=cbar_ax)
    # Save figure
    plt.savefig(os.path.join(simulator.saving_dir, "results.png"))

    plt.figure(figsize=(10, 10))
    plt.hist(
        simulator.sensor.screen.flatten(),
        bins=np.arange(0, np.round(np.max(simulator.sensor.screen)), 0.05)
    )
    plt.savefig(os.path.join(simulator.saving_dir, "charge_histogram.png"))

def main():
    """
    Main function of the script. Parses the arguments, runs the simulation, and
    saves the results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--parallelize", action="store_true", help="Enable parallelization"
    )
    args = parser.parse_args()

    source, slit, sensor, decoder, options = get_objects_from_config(config_path=args.config)
    simulator = CodApSimulator(source=source, slit=slit, sensor=sensor, options=options, decoder=decoder)

    play_simulation(simulator=simulator, config_path=args.config, parallelize=args.parallelize)
    plot_results(simulator)

if __name__ == "__main__":
    main()
