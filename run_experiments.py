import os
import json
from utils import get_config_name
import numpy as np
import codedapertures as ca

def run_sensor_comparison_experiment(config_path: str):
    config_name = get_config_name(config_path)
    sensors = ['normal_ccd', 'skipper', 'perfect']
    for sensor in sensors:
        # Load the config file and modify the "sensor" and "name" fields
        with open(config_path, 'r') as f:
            config = json.load(f)
        split_config_name = config['options']['name'].split(' | ')
        config['sensor']['type'] = sensor
        config['options']['name'] = f"{split_config_name[0]} | {split_config_name[1]} | {sensor}"
        # Save the modified config file
        with open(f'modified_configs/{config_name}.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/fourier_decoding_experiment.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/{config_name}.json')

def run_small_to_large_experiment(config_path: str):
    """
    This experiment is made to observe the differences on the image reconstruction quality for
    a simple pattern centered on a source, varying their size.
    """

def run_mura_slit_size_experiment(config_path: str, mura_rank_list: list[int]):
    """
    In this experiment, we observe the difference on the image reconstruction quality for
    different slit sizes. Specifically, mura pattern slits.
    """
    # Name the config as the experiment.
    config_name = 'mura_slit_size_experiment'
    for mura_rank in mura_rank_list:

        # Get the shape of the new slit
        mura = ca.mura(rank=mura_rank)
        mask_size = list(mura.aperture.shape)

        with open(config_path, 'r') as f:
            config = json.load(f)
        split_name = config['options']['name'].split(' | ')
        config['slit']['mura_config']['rank'] = int(mura_rank)
        config['slit']['mask_size'] = mask_size
        config['options']['name'] = f'{split_name[0]} | mura {mura_rank} | {split_name[2]}'
        with open(f'modified_configs/{config_name}.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/{config_name}.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/{config_name}.json')

def run_field_of_view_experiment(config_path: str, field_of_view_list: list[float]):
    """
    
    """
    # Name the config as the experiment.
    config_name = 'field_of_view_experiment'
    for field_of_view in field_of_view_list:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['options']['field_of_view']['automatically_set_slit_to_sensor_distance'] = True
        config['options']['field_of_view']['fully_coded_field_of_view'] = field_of_view
        config['options']['name'] = f"{config['options']['name']} | {field_of_view}"
        with open(f'modified_configs/{config_name}.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/{config_name}.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/{config_name}.json')

def run_sensor_resolution_change_experiment(config_path: str, sensor_resolutions: list[list[int]]):

    # Name the config as the experiment.
    config_name = 'sensor_resolution_change_experiment'
    for sensor_resolution in sensor_resolutions:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['sensor']['mask_resolution'] = sensor_resolution
        config['options']['name'] = f"{config['options']['name']} | {sensor_resolution}"
        with open(f'modified_configs/{config_name}.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/{config_name}.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/{config_name}.json')

def run_reconstruction_method_experiment(config_path: str):

    # List available reconstruction methods
    # reconstruction_methods = ['fourier', 'mura', 'general']
    reconstruction_methods = ['fourier', 'general']

    # Name the config as the experiment.
    config_name = 'reconstruction_method_experiment'
    for method in reconstruction_methods:
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['decoder']['method'] = method
        config['options']['name'] = f"{config['options']['name']} | {method}"
        with open(f'modified_configs/{config_name}.json', 'w') as f:
            json.dump(config, f, indent=4)

        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/{config_name}.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/{config_name}.json')

if __name__ == '__main__':
    # CONFIG_PATH = 'configs/mura_experiment.json'
    # run_mura_slit_size_experiment(CONFIG_PATH, mura_rank_list=np.arange(1, 9, 1))
    # run_field_of_view_experiment(CONFIG_PATH, field_of_view_list=np.linspace(1, 40, 10))
    # CONFIG_PATH = 'configs/resolution_testing.json'
    # run_sensor_resolution_change_experiment(CONFIG_PATH, sensor_resolutions=[[200, 200], [400, 200], [600, 200], [800, 200], [1000, 200]])
    # CONFIG_PATH = 'config.json'
    # run_reconstruction_method_experiment(CONFIG_PATH)
    CONFIG_PATH = 'config.json'
    run_reconstruction_method_experiment(CONFIG_PATH)
