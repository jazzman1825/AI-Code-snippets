import os
import cv2
import numpy as np
import pickle

# Load the image tensors
opetus_ja_vastetiedot = pickle.load(open("ImageSegmentation/Data/opetustiedot.p", "rb"))

# Get the tensors
syotekuva_tensori = opetus_ja_vastetiedot[0]
vastekuva_tensori = opetus_ja_vastetiedot[1]

# Output directories for saving images
syotekuva_directory = "/Users/mori/Documents/Koulu/AI-Project/Code Snippets/ImageSegmentation/Data/Images"
vastekuva_directory = "/Users/mori/Documents/Koulu/AI-Project/Code Snippets/ImageSegmentation/Data/Masks"

# Create the output directories if it doesn't exist
os.makedirs(syotekuva_directory, exist_ok=True)
os.makedirs(vastekuva_directory, exist_ok=True)

# Iterate through each image and save it
for i in range(syotekuva_tensori.shape[0]):
    # Extract the 2D image array from the tensor
    syotekuva_image = syotekuva_tensori[i, :, :]
    vastekuva_image = vastekuva_tensori[i, :, :]

    # Create file paths for saving images
    syotekuva_path = os.path.join(syotekuva_directory, f"syotekuva_{i}.jpeg")
    vastekuva_path = os.path.join(vastekuva_directory, f"vastekuva_{i}.png")

    # Save the images
    cv2.imwrite(syotekuva_path, syotekuva_image)
    cv2.imwrite(vastekuva_path, vastekuva_image)
