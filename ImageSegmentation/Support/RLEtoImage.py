import numpy as np
from PIL import Image

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

# Example usage
rle_string = "353793 64 354049 192 354305 6912 361345 128"
image_shape = (1600, 256)

image = create_image_from_rle(rle_string, image_shape)
image.show()
