import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.layers import Layer

# -----------------------------
# Placeholder Cast layer for deserialization
# -----------------------------
class Cast(Layer):
    def __init__(self, dtype='float32', **kwargs):
        super().__init__(**kwargs)
        self._dtype = dtype

    def call(self, inputs):
        return tf.cast(inputs, self._dtype)

# -----------------------------
# Load the model safely
# -----------------------------
model = tf.keras.models.load_model(
    "FER_25_VGG19.keras",
    compile=False,
    custom_objects={"Cast": Cast}
)

# -----------------------------
# Emotion labels
# -----------------------------
emotion_categories = [
    'Anger', 'Disgust', 'Fear',
    'Happiness', 'Neutral', 'Sadness', 'Surprise'
]

# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(img):
    """
    img: NumPy array from Gradio input
    Returns: dict of emotion probabilities
    """
    if img is None:
        return {emotion: 0.0 for emotion in emotion_categories}

    # Convert to tensor and float32
    img = tf.convert_to_tensor(img)
    img = tf.cast(img, tf.float32)

    # Ensure 3 channels (RGB)
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)

    # Resize to model input size
    img = tf.image.resize(img, (224, 224))

    # Add batch dimension
    img_array = tf.expand_dims(img, axis=0)

    # Preprocess for VGG19
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)[0]

    # Return probabilities as dictionary
    return {emotion_categories[i]: float(predictions[i]) for i in range(len(emotion_categories))}

# -----------------------------
# Gradio Interface
# -----------------------------
iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=3),
    title="Facial Emotion Recognition (VGG19)",
    description="Upload a face image and the model will predict the emotion."
)

# -----------------------------
# Launch
# -----------------------------
if __name__ == "__main__":
    iface.launch()