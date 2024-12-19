from normalization import normalize_dataset
import os
import tensorflow as tf
from keras import layers, models , applications

dataser_dir = "data"

# check if data exists in the directory
if not os.path.exists(dataser_dir):
    print("Data directory does not exist")
    exit()

normalize_dataset(data_dir=dataser_dir)



def preprocess_dataset(image, label):
    image = applications.resnet.preprocess_input(image)  # Apply ResNet-specific preprocessing
    return image, label


# Path to the dataset
dataset_path = "normalized_data/"

# Parameters
image_size = (224, 224)  # Resize images to 224x224
batch_size = 32          # Number of images per batch
seed = 123               # Seed for reproducibility

# Create the training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,  # Reserve 20% of data for validation
    subset="training",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)
train_ds = train_ds.map(preprocess_dataset)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=image_size,
    batch_size=batch_size
)
val_ds = val_ds.map(preprocess_dataset)

# Data augmentation
data_augmentation_layer = tf.keras.Sequential([                                    
  tf.keras.layers.RandomFlip('horizontal_and_vertical'),
  tf.keras.layers.RandomRotation(1.0),
  tf.keras.layers.RandomZoom(0.3),
  tf.keras.layers.RandomContrast(0.3)                  
], name='data_augmentation')

# Load ResNet50 as a feature extractor
resnet_base = tf.keras.applications.ResNet50(
    include_top=False,  # Exclude the dense layers
    weights='imagenet', # Use pretrained weights
    input_shape=(224, 224, 3)  # Input image shape
)

# Freeze the base model
resnet_base.trainable = False

# Create the classification head
fcn = models.Sequential([
    layers.GlobalAveragePooling2D(),  # Pool the spatial dimensions
    layers.Dense(128, activation='relu'),  # Add a dense layer with 128 units
    layers.Dropout(0.5),  # Regularization to prevent overfitting
    layers.Dense(2, activation='softmax')  # Output layer for 2 classes
])

# Build the complete model
model = models.Sequential([
    data_augmentation_layer,
    resnet_base,
    fcn
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',  # Use categorical_crossentropy for one-hot labels
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# print the accuracy of the model
print(model.evaluate(val_ds))

# Save the model
model.save('model.keras')