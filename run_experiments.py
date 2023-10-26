import os
import json
from utils import get_config_name

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
        with open(f'modified_configs/{config_name}', 'w') as f:
            json.dump(config, f, indent=4)
        
        # Run the experiment with the modified config file
        os.system(f'python experiment.py --config modified_configs/fourier_decoding_experiment.json --parallelize')
        # Remove the modified config file
        os.remove(f'modified_configs/fourier_decoding_experiment.json')

def run_small_to_large_experiment(config_path: str):
    """
    This experiment is made to observe the differences on the image reconstruction quality for
    a simple pattern centered on a source, varying their size.
    """

if __name__ == '__main__':
    config_path = 
    run_sensor_comparison_experiment(config_path)