import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical  # Correct import for to_categorical

# Function to load data
def load_data(image_dir, mask_dir, target_size=(128, 128), num_classes=2):
    images = []
    masks = []

    # Load images and masks
    for img_name in os.listdir(image_dir):  # Fixed typo in os.listdir
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        # Load and preprocess image
        img = Image.open(img_path).resize(target_size)
        img = np.array(img) / 255.0  # Normalize image values to [0, 1]

        # Load and preprocess mask
        mask = Image.open(mask_path).resize(target_size)  # Fixed variable name typo
        mask = np.array(mask)
        mask = to_categorical(mask, num_classes=num_classes)  # One-hot encoding for segmentation

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)  # Fixed missing comma


# Generator function for loading data in batches
def data_generator(image_dir, mask_dir, batch_size=32, target_size=(128, 128), num_classes=2):
    while True:
        images, masks = [], []
        img_names = os.listdir(image_dir)  # List of image names

        for _ in range(batch_size):
            # Randomly sample an image and corresponding mask
            img_name = np.random.choice(img_names)  # Fixed np.random.choices to np.random.choice
            img_path = os.path.join(image_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)

            # Load and preprocess image
            img = Image.open(img_path).resize(target_size)
            img = np.array(img) / 255.0

            # Load and preprocess mask
            mask = Image.open(mask_path).resize(target_size)
            mask = np.array(mask)
            mask = to_categorical(mask, num_classes=num_classes)

            images.append(img)
            masks.append(mask)

        yield np.array(images), np.array(masks)  # Yield batch of images and masks
