import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_cv
import os
import matplotlib.pyplot as plt 


image_size = (180, 180)
img = keras.utils.load_img("PetImages/Cat/0.jpg", target_size=image_size)
plt.imshow(img)
plt.show()
