import json
import numpy as np
from tqdm import tqdm
import os
import shutil
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import timeit

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
        elif "pattern" in self.mask_type:
            return self.load_pattern_slit()
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

    def load_pattern_slit(self):
        path = f"patterns/{self.mask_type}.png"
        # Load the png image as a mask matrix
        try:
            mask = plt.imread(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File {path} not found. Check if it's correct"
            ) from exc
        mask = mask[:, :, 3]
        # Resize the mask to the desired size
        mask = np.resize(mask, self.mask_size)
        plt.show()
        return mask


@dataclass
class SourceScreen:
    mask_size: list
    mask_type: str
    mask_width: int
    photons_per_pixel: int

    def __post_init__(self) -> None:

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

        self.saving_dir = self.get_path_to_save()
        self.rng = np.random.default_rng(seed=options.random_seed)

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
        self.sensor.noise_matrix += np.round(
            self.rng.normal(
                loc=0, scale=self.sensor.readout_noise, size=self.sensor.screen.shape
            )
        )
        self.sensor.screen += self.sensor.noise_matrix

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

    def passes_through_slit(self, positions, angles):
        """
        Returns a boolean array indicating whether each photon passes through the slit.

        Parameters:
        positions: np.array
            The positions of the pixels in the source mask for all photons.
        angles: np.array
            The angles of the director vectors for all photons.

        Returns:
        np.array
            A boolean array where True indicates a photon passes through the slit.
        """

        # Compute the intersection pixel coordinates for all photons
        intersection_pixel_coordinates = self.compute_landing_pixels(
            positions, angles, self.options.source_to_slit_distance
        )

        # Check if the intersection coordinates are within the slit dimensions for all photons
        within_slit_bounds = (
            (intersection_pixel_coordinates[:, 0] >= 0)
            & (intersection_pixel_coordinates[:, 0] < self.slit.mask.shape[0])
            & (intersection_pixel_coordinates[:, 1] >= 0)
            & (intersection_pixel_coordinates[:, 1] < self.slit.mask.shape[1])
        )

        # Create a boolean array of the same length as positions, initialized with False
        passes_through = np.zeros(positions.shape[0], dtype=bool)

        # Set True for the photons that are within slit bounds and the corresponding
        # slit.mask value is 1
        valid_indices = np.where(within_slit_bounds)[0]
        valid_coords = intersection_pixel_coordinates[valid_indices]
        passes_through[valid_indices] = (
            self.slit.mask[valid_coords[:, 0], valid_coords[:, 1]] == 1
        )

        return passes_through

    def compute_landing_pixels(self, positions, angles, z_distance):
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

        # Create the origin pixel vectors for all pixels.
        origin_vectors = np.column_stack(
            (
                positions[:, 0] * self.options.inter_pixel_distance,
                positions[:, 1] * self.options.inter_pixel_distance,
                positions[:, 2],
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
        intersection_pixel_coordinates = (
            intersection_vectors[:, :2] / self.options.inter_pixel_distance
        ).astype(int)
        return intersection_pixel_coordinates

    def sample_angles(self):
        """Samples the angles of the photons from a uniform distribution"""
        theta = self.rng.uniform(*self.options.theta_bounds, self.source.mask.shape)
        phi = self.rng.uniform(*self.options.phi_bounds, self.source.mask.shape)
        return theta, phi


def play_simulation(simulator: CodApSimulator, config_path: str):
    """Plays the simulation"""
    print("Simulating the propagation of photons through the slit...")

    tic = timeit.default_timer()
    simulator.make_image()
    toc = timeit.default_timer()
    print(f"Done! time: {toc-tic: .2f}s")
    if simulator.options.add_noise:
        print("Adding noise to the image...")
        tic = timeit.default_timer()
        simulator.add_noise()
        toc = timeit.default_timer()
        print(f"Done! time: {toc-tic: .2f}s")
    print("Done!")
    print("Saving results...")
    tic = timeit.default_timer()
    simulator.save_results(config_path)
    toc = timeit.default_timer()
    print(f"Done! time: {toc-tic: .2f}s")

def main():
    """Main function of the script. Parses the arguments and runs the simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    config_path = args.config
    source, slit, sensor, options = get_objects_from_config(config_path=config_path)
    simulator = CodApSimulator(source=source, slit=slit, sensor=sensor, options=options)
    play_simulation(simulator=simulator, config_path=config_path)

    # Plot the results
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(1, 3, 1)
    vmin, vmax = 0, float(np.max(simulator.source.screen))
    im = plt.imshow(simulator.source.screen, vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.05, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.title("Source Photons")
    plt.subplot(1, 3, 2)
    plt.imshow(simulator.slit.mask, cmap="binary_r")
    plt.title("Slit screen")
    plt.subplot(1, 3, 3)
    vmin, vmax = 0, float(np.max(simulator.sensor.screen))
    im = plt.imshow(simulator.sensor.screen, vmin=vmin, vmax=vmax)
    plt.imshow(simulator.sensor.screen, vmin=vmin, vmax=vmax)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.title("Detected Photons")
    # Save figure
    plt.savefig(os.path.join(simulator.saving_dir, "results.png"))

    # plot the noise matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(simulator.sensor.noise_matrix)
    plt.colorbar()
    plt.savefig(os.path.join(simulator.saving_dir, "noise_matrix.png"))


if __name__ == "__main__":
    main()
