"""
This script is intended for the pre-processing of the png images used to build the
source and slit masks for the simulation. The images are first converted to grayscale,
and then reshaped to the desired size. The images are then saved as numpy arrays.
"""

import cv2
