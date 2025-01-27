import numpy as np
from sklearn.model_selection import train_test_split
from utils import LoadData, PreprocessData
from models.model import UNetCompiled
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os

print(tf.__version__)

# -------------------------------------------------------------------
# 0. Configure and Verify GPU Availability
# -------------------------------------------------------------------
print("Configuring TensorFlow to use GPU if available...")

print("TensorFlow Version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPUs Detected:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        
        # Enable memory growth to allocate GPU memory as needed
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is set for TensorFlow: {gpus}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. TensorFlow will use the CPU.")

# -------------------------------------------------------------------
# 1. Load and Preprocess Data
# -------------------------------------------------------------------
print("\nLoading and preprocessing data...")
image_dir = r'c:/Users/kiril/Desktop/oxford-iiit-pet/images'
mask_dir = r'C:/Users/kiril/Desktop/oxford-iiit-pet/annotations/trimaps'

# Load image and mask filenames
img, mask = LoadData(image_dir, mask_dir)

# Preprocess images and masks
X, y = PreprocessData(
    img, mask,
    target_shape_img=(128, 128, 3),
    target_shape_mask=(128, 128, 1),
    path1=image_dir,
    path2=mask_dir
)

# -------------------------------------------------------------------
# 2. Train/Validation Split
# -------------------------------------------------------------------
print("\nSplitting data into training and validation sets...")
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Validation set size: {X_valid.shape[0]} samples")

# Modify training setup
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\
    .batch(BATCH_SIZE)\
    .prefetch(AUTOTUNE)
valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))\
    .batch(BATCH_SIZE)\
    .prefetch(AUTOTUNE)

# -------------------------------------------------------------------
# 3. Initialize the Model
print("\nInitializing the U-Net model...")
unet = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3)

# After model initialization and weights loading
print("\nCompiling model...")
unet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# -------------------------------------------------------------------
# 4. Load Pre-trained Weights

# Define weights path
weights_path = 'unet_weights.weights.h5'
full_weights_path = os.path.join(os.path.dirname(__file__), weights_path)

# Check if weights exist
if not os.path.exists(full_weights_path):
    print(f"Error: Weights file not found at {full_weights_path}")
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir())
    exit(1)

# Load weights with verified path
print("\nLoading pre-trained weights...")
unet.load_weights(full_weights_path)
print("Weights loaded successfully")

# -------------------------------------------------------------------
# 5. Evaluate Model
print("\nEvaluating model...")
evaluation = unet.evaluate(valid_dataset)
print(f"Validation Loss: {evaluation[0]:.4f}")
print(f"Validation Accuracy: {evaluation[1]:.4f}")

# -------------------------------------------------------------------
# 6. Make Predictions
print("\nGenerating predictions on validation set...")
predictions = unet.predict(valid_dataset)

# -------------------------------------------------------------------
# 7. Visualize Results
def VisualizeResults(index):
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = unet.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index,:,:,0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:,:,0])
    arr[2].set_title('Predicted Masked Image ')
    plt.show()

print("\nVisualizing results...")
# Visualize first 3 images from validation set
for i in range(3):
    print(f"\nVisualizing result {i+1}/3")
    VisualizeResults(i)

# -------------------------------------------------------------------
# 8. Save the Final Model Weights
# -------------------------------------------------------------------
print("\nSaving final model weights...")
unet.save_weights('unet_weights_final.weights.h5')
print("Weights saved to 'unet_weights_final.weights.h5'.")

# Save entire model in Keras format
unet.save('unet_model_final.keras')  # Changed from .h5 to .keras
print("Entire model saved to 'unet_model_final.keras'.")
