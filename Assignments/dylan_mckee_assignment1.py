# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 22:18:10 2024

@author: Dylan
"""

#%% Importing 

import numpy as np
import pathlib
import glob
import os
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg


#%% Sorted List

fdir = 'C:/Users/rick/OneDrive/Documents/PSY 627 Data/fLoc_stimuli'

image_extensions = ('.jpg', '.jpeg')

#List of image files in the folder filtered by .jpg
image_files = [file for file in os.listdir(fdir) if file.lower().endswith(image_extensions)]

# Sort the image files by name
sorted_image_files = sorted(image_files)

# It worked!
for image in sorted_image_files:
    print(image)
    
#%% Random Sample of 12 images in Grid

# Randomly sample 12 images from the list
sampled_images = random.sample(image_files, 12)

# Set up the figure and axes for displaying images in a 3x4 grid
fig, axes = plt.subplots(3, 4, figsize=(12, 9))

# Loop through the sampled images and display each one
for i, img_file in enumerate(sampled_images):
    # Load the image
    img_path = os.path.join(fdir, img_file)
    img = mpimg.imread(img_path)

    # Convert the image to greyscale if it is a colored image (assumes RGB format)
    if len(img.shape) == 3:  # Check if it's a 3-channel image (RGB)
        img_gray = np.dot(img[..., :3], [0.2989, 0.587, 0.114])  # RGB to grayscale conversion
    else:
        img_gray = img  # If it's already a grayscale image

    # Get the axis for the current image
    ax = axes[i // 4, i % 4]

    # Display the grayscale image
    ax.imshow(img_gray, cmap='gray')
    ax.axis('off')  # Turn off axis labels

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the figure
plt.show()


#%% Random Sample in Sequential Format

sampled_images = random.sample(image_files, min(len(image_files), 12))
# Determine the number of rows and columns
n_images = 12
# Set up the figure
n_cols = 12
n_rows = 1

# Set up the figure and axes for displaying images in a grid
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows))

# Loop through the subplots and display a randomly chosen image in each subplot
for ax in axes:
    # Randomly select an image from the sampled images
    random_image = random.choice(sampled_images)

    # Load the image
    img_path = os.path.join(fdir,random_image)
    img = mpimg.imread(img_path)

    # Display the image in subplots
    ax.imshow(img)
    ax.axis('off')  

# Show
plt.show()


#%% Save as array
image_array_list = []

# Loop through the sampled images and load them into the list as NumPy arrays
for img_file in sampled_images:
    # Load the image
    img_path = os.path.join(fdir, img_file)
    img = mpimg.imread(img_path)
image_array_list.append(img)

# Stack the images into a single NumPy array (assuming all images are the same size)
# If they are not the same size, additional resizing steps will be needed
image_array = np.stack(image_array_list)


np.save('randomly_selected-images', image_array)

print("Array saved as 'randomly_selected-images.npy'")


#%%
loaded_images = np.load('randomly_selected-images.npy')
print(loaded_images.shape)  # Check the shape of the array