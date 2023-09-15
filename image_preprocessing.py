"""
This script is intended for the pre-processing of the png images used to build the
source and slit masks for the simulation. The images are first converted to grayscale,
and then reshaped to the desired size. The images are then saved as numpy arrays.
"""

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def process_image(file_name, type = 'png', target_size=(100, 100), threshold = 90, invert = False):
    try:
        # Open the image using Pillow
        image = Image.open(f'patterns/{file_name}.{type}')

        # Resize the image to the specified dimensions
        image = image.resize(target_size, Image.ANTIALIAS)

        # Convert the image to grayscale
        image = image.convert('L')

        # Convert the image to a NumPy array
        image_array = np.array(image)

        #Generate Binary Image
        binary_image = (image_array > threshold).astype(np.uint8)

        #Invert the image if necesary 
        if invert == True:
            binary_image = 1 - binary_image

        #plot both images to check if the threshold is good enough
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 2, 1)
        plt.imshow(image_array, cmap = 'gray')
        plt.title("Image Array")
        plt.subplot(2, 2, 2)
        plt.imshow(binary_image, cmap='gray') 
        plt.title(f"Binary Image - Threshold: {threshold}")
        plt.show()

        binary_image_sv = Image.fromarray((binary_image*255).astype(np.uint8))
        binary_image_sv.save(f'patterns/{file_name}_pattern.png')

        return image_array, binary_image

    except Exception as e:
        print(f"Error processing image {file_name}: {e}")
        return None
