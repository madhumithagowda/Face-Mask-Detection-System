import streamlit as st
import numpy as np
from PIL import Image


# Rebuild the model architecture
@st.cache_resource
def load_facemask_model():
    """Load face mask model by rebuilding architecture and loading weights"""
    
    # Use MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    # Rebuild the exact same architecture
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(128, 128, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Load the weights
    model.load_weights('facemask_weights.weights.h5')
    st.success("✅ Face Mask Model loaded successfully!")
    return model

# Page config
st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")

st.title("😷 Face Mask Detection System")
st.markdown("Upload an image of a person to detect if they are **wearing a mask** or **not wearing a mask**.")
st.markdown("---")

# Load model
model = load_facemask_model()

IMG_SIZE = 128

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    if st.button("🔍 Detect Mask", use_container_width=True):
        with st.spinner("Analyzing image..."):
            # Preprocess image
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = prediction[0][0]
            
            # Determine class (SWAPPED - fixed reverse detection)
            if confidence > 0.5:
                predicted_class = "Without Mask"
                confidence_percent = confidence * 100
            else:
                predicted_class = "With Mask" 
                confidence_percent = (1 - confidence) * 100

            st.markdown("---")
            if predicted_class == "With Mask":
                st.success(f"### ✅ {predicted_class}")
                st.success(f"Confidence: **{confidence_percent:.1f}%**")
                st.markdown("Person is **wearing a mask correctly**. Good safety practice!")
            else:
                st.warning(f"### ⚠️ {predicted_class}")
                st.warning(f"Confidence: **{confidence_percent:.1f}%**")
                st.markdown("**No mask detected!** Please wear a mask for safety.")

            st.markdown("---")
            st.markdown("**Prediction Details:**")
            st.progress(int(confidence_percent), text=f"With Mask: {confidence_percent:.1f}%")
            st.progress(int((1-confidence)*100), text=f"Without Mask: {(1-confidence)*100:.1f}%")

st.markdown("---")
st.caption("Built with Streamlit | Face Mask Detection CNN")
