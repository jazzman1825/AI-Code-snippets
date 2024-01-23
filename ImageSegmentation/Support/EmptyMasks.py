import os
import numpy as np
from PIL import Image

def create_empty_mask(image_id, image_shape, output_folder="."):
    filename = f"{output_folder}/{image_id}.jpg"
    mask = np.zeros(image_shape, dtype=np.uint8)
    empty_image = Image.fromarray(mask)
    empty_image.save(filename)
    print(f"Created empty mask for: {filename}")

image_shape = (256, 1600)

# Paths to the directories
original_images_directory = 'ImageSegmentation/Data/severstal-steel-defect-detection/train_images'
mask_images_directory = 'ImageSegmentation/Support/RLEtoMaskAllDefected'

# Get the list of original image files
original_image_files = os.listdir(original_images_directory)

# Assuming the images have the same resolution as before
for original_image in original_image_files:
    image_id, _ = os.path.splitext(original_image)

    # Check if there is a corresponding mask
    mask_filename = f"{mask_images_directory}/{image_id}.jpg"
    
    if not os.path.exists(mask_filename):
        # If there is no corresponding mask, create an empty one
        create_empty_mask(image_id, image_shape, output_folder="ImageSegmentation/Support/AllMasks")
