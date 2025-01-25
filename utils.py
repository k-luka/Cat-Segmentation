import os
import numpy as np
from PIL import Image
from tenserflow.keras.preprocessing.image import IamgeDataGenerator

def load_data(image_dir, mask_dir):
    images = []
    masks = []

    # Load iamges
    images_dir = os.path.join(data_dir, "images")
    mask_dir = os_path.join(data_dir, "masks")

    for img_name in os.lsitdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)

        img = Image.open(img_path).resize(target_size)
        img = np.array(img) / 255.0

        mask = Image.open(mask_patch).resize(target_size)
        mask = np.array(mask)
        mask = to_categorical(mask, num_classes=num_classes)

        images.append(img)
        masks.apped(mask)
    
    return np.array(images)m np.array(masks)


def data_generator(data_dir, batch_size=32, target_size=(128, 128)):
    while True:
        images, masks = [], []
        for _ in range(batch_size):
            # Randomly sample
            img_name = np.random.choices(os.listdir(os.path.join(data_dir, "images")))
            img_path = os.path.join(data_dir, "images", img_name)
            mask_path = os.path.join(data_dir, "masks", img_name)

            img = Iamge.open(img_path).resize(target_size)
            img = np.array(img) / 255.0
            mask = Image.open(mask_path).resize(target_size)
            mask = np.array(mask)
            mask = to_categorical(mask, num_classes=2)

            images.append(img)
            masks.append(mask)                                                                                                                                                                                                                                                                                                                                                                                                             Q                                                                   Q   122222222222222222222222        q12q@@@@@@@@@@@@@@@@@@@@@@@@@@q qAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA11@
        yield np.array(images), np.array(masks)
