import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Function to load data
def load_data(dataset_path):
    images = []
    labels = []
    label_map = {}  # Mapping of original label to a continuous range starting from 0
    label_counter = 0
    
    # Traverse each subdirectory (which is the class label)
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            # Process both 'left' and 'right' images in each label subdirectory
            for side in ['left', 'right']:
                side_path = os.path.join(label_path, side)
                if os.path.isdir(side_path):
                    for img_name in os.listdir(side_path):
                        img_path = os.path.join(side_path, img_name)
                        if img_name.lower().endswith('.bmp'):  # Check for BMP files
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = cv2.resize(img, (224, 224))  # Resize to standard size
                                images.append(img)
                                # Map original label to continuous labels
                                if label not in label_map:
                                    label_map[label] = label_counter
                                    label_counter += 1
                                labels.append(label_map[label])  # Use the mapped label
    
    return np.array(images), np.array(labels), len(label_map)

# Example usage:
dataset_path = "C:/Users/Darshan/OneDrive/Desktop/FingerVeinDetection/FingerVeinDetection/archive/data"
images, labels, num_classes = load_data(dataset_path)

# Check if images and labels were loaded correctly
print(f"Total images: {len(images)}")
print(f"Total labels: {len(labels)}")

if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images or labels found. Please check your dataset path and structure.")

# One-hot encode labels
labels = to_categorical(labels, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape images for input into a neural network (channels last)
X_train = X_train.reshape(-1, 224, 224, 1).astype('float32') / 255.0
X_val = X_val.reshape(-1, 224, 224, 1).astype('float32') / 255.0

# Define the model
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save("finger_vein_model.h5")
