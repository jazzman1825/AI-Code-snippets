import numpy as np
import pandas as pd
import os
from PIL import Image

def create_or_update_image(image_id, encoded_pixels, image_shape, output_folder="."):
    filename = f"{output_folder}/{image_id}"
    
    if os.path.exists(filename):
        # If the image already exists, load and update it
        existing_image = Image.open(filename)
        mask = rle_decode(encoded_pixels, image_shape)
        existing_array = np.array(existing_image)
        updated_array = np.maximum(existing_array, mask)
        updated_image = Image.fromarray(updated_array)
        updated_image.save(filename)
        print(f"Updated existing image: {filename}")
    else:
        # If the image doesn't exist, create a new one
        mask = rle_decode(encoded_pixels, image_shape)
        new_image = Image.fromarray(mask)
        new_image.save(filename)
        print(f"Created new image: {filename}")

def rle_decode(rle_string, image_shape):
    width, height = image_shape
    mask = np.zeros(width * height, dtype=np.uint8)
    rle_list = list(map(int, rle_string.split()))

    for i in range(0, len(rle_list), 2):
        start = rle_list[i] - 1
        length = rle_list[i + 1]
        mask[start:start + length] = 255

    return mask.reshape((height, width), order='F')

def create_image_from_rle(encoded_string, image_shape):
    mask = rle_decode(encoded_string, image_shape)
    image = Image.fromarray(mask)
    return image

def save_image_with_id(image, image_id, output_folder="."):
    filename = f"{output_folder}/{image_id}"
    image.save(filename)
    print(f"Image saved as {filename}")

# Example usage
# rle_string = "353793 64 354049 192 354305 6912 361345 128"
image_shape = (1600, 256)

# image = create_image_from_rle(rle_string, image_shape)
# image.show()
# image_id = "c878411d7"

# image = create_image_from_rle(rle_string, image_shape)
# save_image_with_id(image, image_id, output_folder="ImageSegmentation/Support/RLEtoImageTest")
    
csv_file_path = '/Users/mori/Documents/Koulu/AI-Project/severstal-steel-defect-detection/train.csv'
df = pd.read_csv(csv_file_path)

for index, row in df.iterrows():
    image_id = row['ImageId']
    encoded_pixels = row['EncodedPixels']
    
    # Create or update image from RLE
    create_or_update_image(image_id, encoded_pixels, image_shape, output_folder="ImageSegmentation/Support/RLEtoImageTest")