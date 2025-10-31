'''
Find the Saliency threshold for UTOM
'''

import numpy as np
from PIL import Image
from skimage import filters
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path_A', type=str, help='Path to one input image')
    parser.add_argument('--image_path_B', type=str, help='Path to one target image')
    args = parser.parse_args()
    img = Image.open(args.image_path_A)
    threshold_A = filters.threshold_otsu(np.asarray(img))
    print(f"Computed Watershed Threshold for Image A: {threshold_A}")

    img = Image.open(args.image_path_B)
    threshold_B = filters.threshold_otsu(np.asarray(img))
    print(f"Computed Watershed Threshold for Image B: {threshold_B}")

