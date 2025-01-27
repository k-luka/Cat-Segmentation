import numpy as np
from sklearn.model_selection import train_test_split
from utils import LoadData, PreprocessData
from models.model import UNetCompiled
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

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
image_dir = r'/content/drive/MyDrive/images'
mask_dir = r'/content/drive/MyDrive/annotations/trimaps'

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

# -------------------------------------------------------------------
# 3. Initialize the Model
# -------------------------------------------------------------------
print("\nInitializing the U-Net model...")
unet = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3)
unet.summary()

# -------------------------------------------------------------------
# 4. Compile the Model
# -------------------------------------------------------------------
print("\nCompiling the model...")
unet.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# -------------------------------------------------------------------
# 5. Define Callbacks
# -------------------------------------------------------------------
print("\nSetting up ModelCheckpoint callback...")

checkpoint = ModelCheckpoint(
    'best_unet_model.h5',       # Filepath to save the model
    monitor='val_loss',         # Metric to monitor
    verbose=1,                  # Verbosity mode
    save_best_only=True,        # Save only the best model
    mode='min'                  # Mode for the monitored metric
)

# -------------------------------------------------------------------
# 6. Train the Model with Interrupt Handling
# -------------------------------------------------------------------
print("\nStarting training...")
try:
    history = unet.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=20,
        batch_size=32,
        callbacks=[checkpoint]
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving current weights...")
    unet.save_weights('unet_weights_interrupted.h5')
    print("Weights saved to 'unet_weights_interrupted.h5'. Exiting training.")

# -------------------------------------------------------------------
# 7. Plot Training History
# -------------------------------------------------------------------
def plot_training_history(history):
    print("\nPlotting training history...")
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    
    # Plot Loss
    axis[0].plot(history.history["loss"], color='r', label='Train Loss')
    axis[0].plot(history.history["val_loss"], color='b', label='Validation Loss')
    axis[0].set_title('Loss Comparison')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend()
    
    # Plot Accuracy
    axis[1].plot(history.history["accuracy"], color='r', label='Train Accuracy')
    axis[1].plot(history.history["val_accuracy"], color='b', label='Validation Accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].set_xlabel('Epoch')
    axis[1].set_ylabel('Accuracy')
    axis[1].legend()
    
    plt.show()

# Only plot if training was completed without interruption
if 'history' in locals():
    plot_training_history(history)

# -------------------------------------------------------------------
# 8. Save the Final Model Weights
# -------------------------------------------------------------------
print("\nSaving final model weights...")
unet.save_weights('unet_weights_final.h5')
print("Weights saved to 'unet_weights_final.h5'.")

# Optional: Save the entire model (architecture + weights)
# unet.save('unet_model_final.h5')
# print("Entire model saved to 'unet_model_final.h5'.")
