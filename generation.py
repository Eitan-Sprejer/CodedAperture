import os
import json
import numpy as np

sources_folder = "training_sources"

def generate_source_masks(
        shape: list,
        n_per_density: int,
        densities: list,
):
    """
    Generates N random source masks of shape shape.
    """
    # Create the training_sources folder if it doesn't exist
    if not os.path.exists(sources_folder):
        os.makedirs(sources_folder)
    
    for k, density in enumerate(densities):# Generate N random matrices and save them to the training_sources folder
        for i in range(n_per_density):
            # Generate a random matrix with 1s and 0s, with the specified density of 1s
            source_mask = np.random.choice([0, 1], size=shape, p=[1 - density, density])
            
            # Save the matrix to a file
            filename = f"training_sources_{k * n_per_density + i}.txt"
            filepath = os.path.join(sources_folder, filename)
            np.savetxt(filepath, source_mask, fmt="%d")
    return None

def generate_training_dataset(
        config_path: str,
        n_per_density: int,
        densities: list,
):
    # Load the config file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Generate the source masks
    generate_source_masks(
        shape=config["source"]["mask_size"],
        n_per_density=n_per_density,
        densities=densities,
    )
    n_masks = len(densities) * n_per_density
    for i in range(n_masks):
        # Modify the "mask_type" of the "source" to the current matrix
        config["source"]["mask_type"] = f"training_sources_{i}.txt"
        config["options"]["name"] = f"{config['options']['name'][:-2]}_{i}"
        
        # Save the modified config file, with the same format
        with open(f"{config_path}_{i}.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Run the experiment.py script with the modified config file
        os.system(f"python experiment.py --config {config_path}_{i}.json --parallelize")
    return None

def main():
    # Generate the training dataset
    generate_training_dataset(
        config_path="configs/autoencoder_experiment_2.json",
        n_per_density=30,
        densities=[0.2, 0.5, 0.8],
    )
    return None
if __name__ == '__main__':
    main()