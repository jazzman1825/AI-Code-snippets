import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import pickle

from tensorflow import keras

opetus_ja_vastetiedot = pickle.load( open( "ImageSegmentation/Data/opetustiedot.p", "rb" ) )

masks = opetus_ja_vastetiedot[1]

non_zero_masks_indices = []
count = 0

for i, mask in enumerate(masks):
    # Check if the mask contains non-zero values
    if np.any(mask != 0):
        non_zero_masks_indices.append(i)
        count += 1

print("Indices of masks with non-zero values:", non_zero_masks_indices)
print("Number of masks with non-zero values:", count, "Out of", len(masks), "( ≈",  round(count/(len(masks)/100), 2), "% )")
