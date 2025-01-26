import os
import numpy as np
import imageio
from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original images and mask files respectively
    """
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)
    
    orig_img = []
    mask_img = []
    for file in image_dataset:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('._'):
            orig_img.append(file)
    for file in mask_dataset:
        if file.lower().endswith('.png') and not file.startswith('._'):
            mask_img.append(file)
    
    orig_img.sort()
    mask_img.sort()
    
    return orig_img, mask_img

def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes images and masks into arrays of desired size.
    The returned mask dataset is single-channel, with classes starting from 0.
    """
    m = len(img)
    i_h, i_w, i_c = target_shape_img
    m_h, m_w, m_c = target_shape_mask
    
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)
    
    for i, file in enumerate(img):
        # Convert image
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h, i_w))
        single_img = np.array(single_img) / 256.0
        X[i] = single_img
        
        # Convert mask
        single_mask_ind = mask[i]
        mask_path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(mask_path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.array(single_mask)
        single_mask = np.reshape(single_mask, (m_h, m_w, m_c))
        single_mask = single_mask - 1
        y[i] = single_mask
    return X, y

# Example usage:
if __name__ == "__main__":
    image_dir = 'C:/Users/kiril/Desktop/oxford-iiit-pet/images'
    mask_dir = 'C:/Users/kiril/Desktop/oxford-iiit-pet/annotations/trimaps'
    
    img, mask = LoadData(image_dir, mask_dir)
    
    # View examples
    show_images = 5
    for i in range(show_images):
        img_path = os.path.join(image_dir, img[i])
        mask_path = os.path.join(mask_dir, mask[i])
        
        print(f"Attempting to read image: {img_path}")
        print(f"Attempting to read mask: {mask_path}")
        
        try:
            img_view = imageio.imread(img_path)
            mask_view = imageio.imread(mask_path)
            
            print("Image shape:", img_view.shape)
            print("Mask shape:", mask_view.shape)
            
            fig, arr = plt.subplots(1, 2, figsize=(15, 15))
            arr[0].imshow(img_view)
            arr[0].set_title('Image ' + str(i))
            arr[1].imshow(mask_view, cmap='gray')
            arr[1].set_title('Masked Image ' + str(i))
            
            plt.show()
        except Exception as e:
            print(f"Error reading files: {str(e)}")
            print(f"Image file exists: {os.path.exists(img_path)}")
            print(f"Mask file exists: {os.path.exists(mask_path)}")
            try:
                with open(mask_path, 'rb') as f:
                    print(f"First 20 bytes of mask file: {f.read(20)}")
            except Exception as read_error:
                print(f"Error reading mask file content: {str(read_error)}")
    
    # Define the desired shape
    target_shape_img = [128, 128, 3]
    target_shape_mask = [128, 128, 1]

    # Process data using apt helper function
    X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, image_dir, mask_dir)

    # QC the shape of output and classes in output dataset 
    print("X Shape:", X.shape)
    print("Y shape:", y.shape)
    # There are 3 classes : background, pet, outline
    print(np.unique(y))

    # Visualize the output
    image_index = 0
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(X[image_index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y[image_index,:,:,0])
    arr[1].set_title('Processed Masked Image ')

    plt.show()