"""
This script is intended for the pre-processing of the png images used to build the
source and slit masks for the simulation. The images are first converted to grayscale,
and then reshaped to the desired size. The images are then saved as numpy arrays.
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def process_image(file_name, image_type = 'png', target_size=(100, 100), threshold = 90, invert = False):
    """
    This function takes a png image and converts it to a binary image.
    The image is first converted to grayscale, then resized to the desired size.
    The image is then converted to a numpy array and saved as a binary image.

    Parameters
    ----------
    file_name : str
        Name of the file to be processed.
    image_type : str, optional
        Type of the image. The default is 'png'.
    target_size : tuple, optional
        Size of the image after resizing. The default is (100, 100).
    threshold : int, optional
        Threshold value for the binary image. The default is 90.
    invert : bool, optional
        Whether to invert the image or not. The default is False.
    """
    # Open the image using Pillow
    image = Image.open(f'exp_pics/{file_name}.{image_type}')
    # Resize the image to the specified dimensions
    image = image.resize(target_size, Image.ANTIALIAS)

    # Convert the image to grayscale
    image = image.convert('L')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    #Generate Binary Image
    binary_image = (image_array > threshold).astype(np.uint8)

    #Invert the image if necesary
    if invert:
        binary_image = 1 - binary_image

    #plot both images to check if the threshold is good enough
    plt.figure(figsize=(6, 6))
    plt.suptitle('Image Preprocessing')
    plt.subplot(2, 2, 1)
    plt.imshow(image_array, cmap = 'gray')
    plt.title("Image Array")
    plt.subplot(2, 2, 2)
    plt.imshow(binary_image, cmap='gray') 
    plt.title(f"Binary Image - Threshold: {threshold}")
    plt.show()

    # binary_image_to_save = Image.fromarray((binary_image*255).astype(np.uint8))
    # binary_image_to_save.save(f'processed_patterns/{file_name}_processed.png')

    return binary_image
