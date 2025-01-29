import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

# Custom Attention Layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, attention_dim, dropout_rate=0.1, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, 
            key_dim=self.attention_dim, 
            dropout=self.dropout_rate
        )
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(self.dropout_rate)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        attention_output = self.attention(inputs, inputs)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(inputs + attention_output)
        return attention_output

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'attention_dim': self.attention_dim,
            'dropout_rate': self.dropout_rate,
        })
        return config

# Custom Bilinear Pooling Layer
class BilinearPooling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BilinearPooling, self).__init__(**kwargs)

    def call(self, inputs):
        conv1_output, conv2_output = inputs
        conv1_flattened = tf.keras.layers.Flatten()(conv1_output)
        conv2_flattened = tf.keras.layers.Flatten()(conv2_output)
        pooled_output = tf.einsum('ij,ik->ijk', conv1_flattened, conv2_flattened)
        pooled_output = tf.reduce_mean(pooled_output, axis=1)
        return pooled_output

    def get_config(self):
        return super(BilinearPooling, self).get_config()

# Register custom objects
custom_objects = {
    "AttentionLayer": AttentionLayer,
    "BilinearPooling": BilinearPooling,
}

# Load the class labels Excel file
class_labels_df = pd.read_excel('class_labels.xlsx')
class_labels = dict(zip(class_labels_df['Class Index'], class_labels_df['Person Name']))

# Load the model with custom layers
try:
    model = tf.keras.models.load_model('finger_vein_model.h5', custom_objects=custom_objects)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Prediction Function
def predict_finger_vein(image_path):
    if model is None:
        return None, "Model not loaded", None
    if not os.path.isfile(image_path):
        return None, "Image file does not exist", None
    try:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image
        
        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100  # Convert to percentage
        
        # Fetch person name for the predicted class
        person_name = class_labels.get(predicted_class, "Unknown")
        
        # Set person name to "Unknown" if confidence is less than 95%
        if confidence < 95:
            person_name = "Unknown"
        
        return predicted_class, person_name, confidence
    except Exception as e:
        return None, f"Error during prediction: {e}", None

# Main Function
if __name__ == '__main__':
    image_path = input("Enter the image path: ")
    result = predict_finger_vein(image_path)
    
    if result[0] is not None:
        predicted_class, person_name, confidence = result
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Person Name: {person_name}")
        print(f"Matching Confidence: {confidence:.2f}%")
    else:
        print(f"Error: {result[1]}")
