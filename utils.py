import numpy as np
import json
from enum import Enum
from tqdm import tqdm
import os
import shutil
import argparse


def get_config_from_path(path) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
    return config

class MaskGenerator():
    def __init__(self, mask_config: dict):
        self.mask_size = mask_config['mask_size']
        self.mask_width = mask_config['mask_width']
        self.mask_type = mask_config['mask_type']
    def generate_mask(self):
        """Generates a mask with the specified parameters"""
        if self.mask_type == 'phi':
            return self.generate_phi_mask()
        elif self.mask_type == 'pinhole':
            return self.generate_pinhole_mask()
        elif self.mask_type == 'frame':
            return self.generate_frame_mask()
        else:
            raise ValueError('Invalid mask type')
    def generate_phi_mask(self):
        """Generates a mask with a phi shape"""
        height, length = self.mask_size
        width = self.mask_width
        mask = np.zeros((height, length))

        mask[
            int(height*0.1):int(height*0.9),
            int((length-width)/2):int((length+width)/2)
        ] = 1
        mask[
            int(height*0.2):int(height*0.4),
            int((length/3)-(width/2)):int((length/3)+(width/2))
        ] = 1
        mask[
            int(height*0.2):int(height*0.4),
            int((2*length/3)-(width/2)):int((2*length/3)+(width/2))
        ] = 1
        mask[
            int((height/3)-(width/2)):int((height/3)-(width/2)),
            int(length*0.3):int(length*0.7)
        ] = 1
        return mask

    def generate_pinhole_mask(self):
        """Generates a mask with a pinhole shape"""
        height, length = self.mask_size
        mask = np.zeros((height, length))
        hole_radius = int(self.mask_width/2)
        # Define the lambda shape
        mask[
            int(height*0.5)-hole_radius:int(height*0.5)+hole_radius,
            int(length*0.5)-hole_radius:int(length*0.5)+hole_radius
        ] = 1
        return mask

    def generate_frame_mask(self):
        """Generates a mask with a square frame shape"""
        height, length = self.mask_size
        mask = np.zeros((height, length))
        slit_outer_radius = int((np.min(height, length)+self.mask_width)/2)
        mask[
            int(height*0.5)-slit_outer_radius:int(height*0.5)+slit_outer_radius,
            int(length*0.5)-slit_outer_radius:int(length*0.5)+slit_outer_radius
        ] = 1
        slit_inner_radius = int((np.min(height, length)-self.mask_width)/2)
        mask[
            int(height*0.5)-slit_inner_radius:int(height*0.5)+slit_inner_radius,
            int(length*0.5)-slit_inner_radius:int(length*0.5)+slit_inner_radius
        ] = 0
        return mask

class CodApSimulator():
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = get_config_from_path(config_path)
        self.name = self.config['name']
        self.options = self.config['options']
        self.n_photons = self.options['photons_per_pixel']
        self.source2slit_dist = self.options["source_to_slit_distance"]
        self.slit2sensor_dist = self.options["slit_to_sensor_distance"]
        self.source2sensor_dist = self.source2slit_dist + self.slit2sensor_dist
        self.n_photons = self.options["photons_per_pixel"]
        self.pixel_separation = self.options["inter_pixel_distance"]
        
        self.saving_dir = self.get_path_to_save()
        self.rng = np.random.default_rng(seed=self.options['random_seed'])

        self.source_mask_generator = MaskGenerator(self.config['source'])
        self.source_mask = self.source_mask_generator.generate_mask()

        self.slit_mask_generator = MaskGenerator(self.config['slit'])
        self.slit_mask = self.slit_mask_generator.generate_mask()

        self.sensor_screen = np.zeros(self.options['sensor_size'])

    def play_simulation(self):
        print('Simulating the propagation of photons through the slit...')
        self.make_image()
        print('Done!')
        print('Saving results...')
        self.save_results()
        print('Done!')

    def get_path_to_save(self):
        """Returns the path to save the results"""

        os.makedirs('results', exist_ok=True)
        saving_dir = os.path.join('results', self.name)
        os.makedirs(saving_dir, exist_ok=True)
        return saving_dir

    def save_results(self):
        """Saves the results of the simulation"""
        np.save(os.path.join(self.saving_dir, 'image.npy'), self.sensor_screen)
        # Copy the config file to the results folder
        shutil.copy(self.config_path, self.saving_dir)

    def make_image(self):
        """Simulates the propagation of photons through the slit"""

        for _ in tqdm(range(self.n_photons)):
            theta, phi = self.sample_angles()

            for i in range(self.source_mask.shape[0]):
                for j in range(self.source_mask.shape[1]):
                    if self.source_mask[i, j] == 1:
                        passes_through = self.passes_through_slit(
                            np.array([i, j, 0]),
                            (theta[i, j], phi[i, j])
                        )
                        if passes_through:
                            # Compute the position of the photon in the sensor screen.
                            sensor_position = self.compute_landing_pixel(
                                np.array([i, j, self.source2slit_dist]),
                                (theta[i, j], phi[i, j]),
                                self.source2sensor_dist
                            )
                            # Add the photon to the sensor screen.
                            self.sensor_screen[sensor_position[0], sensor_position[1]] += 1

    def passes_through_slit(self, position, angles) -> bool:
        """
        Returns True if the pixel is within the slit, False otherwise.

        Parameters
        ----------
        position: np.array
            The position of the pixel in the source mask.
        angles: tuple
            The angles of the director vector for the photon.
        """
        intersection_pixel_coordinates = self.compute_landing_pixel(
            position,
            angles,
            self.source2slit_dist
        )

        # Check that the intersection is within the slit screen.
        if (np.any(intersection_pixel_coordinates < 0) or
            intersection_pixel_coordinates[0] >= self.slit_mask.shape[0] or
            intersection_pixel_coordinates[1] >= self.slit_mask.shape[1]):
            return False
        else:
            return self.slit_mask[intersection_pixel_coordinates[0], intersection_pixel_coordinates[1]] == 1

    def compute_landing_pixel(self, position, angles, z_distance):
        """
        Computes the pixel in the sensor screen where the photon lands.
        
        Parameters
        ----------
        position: np.array
            The position of the pixel in the source mask.
        angles: tuple
            The angles of the director vector for the photon.
        """
        
        theta, phi = angles[0], angles[1]
        
        # Create the origin pixel vector.
        origin_vector = np.array([*position[:2]*self.pixel_separation, position[2]])
        # Create the direction pixel vector.
        unit_direction_vector = np.array([
            np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)
        ])
        # Compute the intersection of the pixel direction with the slit screen.
        intersection_vector = origin_vector + unit_direction_vector * z_distance \
        / unit_direction_vector[2]
        # Compute the pixel coordinates of the intersection.
        intersection_pixel_coordinates = (intersection_vector[:2] / self.pixel_separation).astype(int)
        return intersection_pixel_coordinates

    def sample_angles(self):
        """Samples the angles of the photons from a uniform distribution"""
        theta = self.rng.uniform(0, 2*np.pi, self.source_mask.shape)
        phi = self.rng.uniform(0, np.pi/2, self.source_mask.shape)
        return theta, phi

def main():
    """Main function of the script. Parses the arguments and runs the simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file')
    args = parser.parse_args()
    config_path = args.config
    simulator = CodApSimulator(config_path)
    simulator.play_simulation()

if __name__ == '__main__':
    main()
