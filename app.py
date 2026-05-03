import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf   # ✅ FIX: added import

# Load model
@st.cache_resource
def load_facemask_model():
    """Load face mask model"""

    # Base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Load weights
    model.load_weights('facemask_weights.weights.h5')

    return model


# UI
st.set_page_config(page_title="Face Mask Detection", page_icon="😷")

st.title("😷 Face Mask Detection System")
st.write("Upload an image to check if a person is wearing a mask.")

# Load model
model = load_facemask_model()

IMG_SIZE = 128

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Detect"):
        img = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        if confidence > 0.5:
            st.error(f"Without Mask ({confidence*100:.2f}%)")
        else:
            st.success(f"With Mask ({(1-confidence)*100:.2f}%)")
