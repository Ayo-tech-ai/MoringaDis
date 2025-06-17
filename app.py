# --------------------------- imports ---------------------------
import streamlit as st

# Page config MUST be before any other Streamlit command
st.set_page_config(page_title="Moringa Leaf Disease Detector", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
from tf_explain.core.grad_cam import GradCAM

# --------------------------- custom CSS ------------------------
st.markdown(
    """
    <style>
    /* Style the main Predict button */
    .stButton>button {
        background-color:#4CAF50;
        color:white;
        font-weight:bold;
        border:none;
        padding:0.6em 1.2em;
        border-radius:6px;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color:#45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------ disease info ------------------------
DISEASE_INFO = {
    "Bacterial Leaf Spot": {...},          # unchanged content
    "Cercospora Leaf Spot": {...},
    "Healthy Leaf": {...},
    "Yellow Leaf": {...},
}

CLASS_NAMES = [
    "Bacterial Leaf Spot",
    "Cercospora Leaf Spot",
    "Healthy Leaf",
    "Yellow Leaf",
]

# ------------------------ load model ------------------------
@st.cache_resource
def load_model():
    m = tf.keras.models.load_model("moringa_effnet_final.keras")
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return m

model = load_model()
backbone = model.get_layer("efficientnetb0")

# ---------------------- grad-cam helper ----------------------
def generate_gradcam(batch, model, class_idx, target_layer="block4c_project_conv"):
    explainer = GradCAM()
    return explainer.explain(
        validation_data=(batch, None),
        model=model,
        class_index=class_idx,
        layer_name=target_layer,
    )

# -------------------------- UI --------------------------
st.title("🌿 Moringa Leaf Disease Detector with Grad-CAM")

uploaded_file = st.file_uploader("Upload a moringa leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Predict"):
        with st.spinner("Running prediction..."):
            # Preprocess
            arr = tf.keras.applications.efficientnet.preprocess_input(
                np.expand_dims(np.array(pil_img, dtype="float32"), axis=0)
            )
            # Predict with full model
            preds = model(arr, training=False)
            class_idx = int(np.argmax(preds))
            class_name = CLASS_NAMES[class_idx]

            st.success(f"🧠 Predicted class: **{class_name}**")

            # Grad-CAM
            heatmap = generate_gradcam(arr, backbone, class_idx)
            st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)

            # Disease info
            info = DISEASE_INFO[class_name]
            with st.expander("🩺 Disease Information", expanded=True):
                st.markdown(f"### {info['name']}")
                st.caption(f"**Cause:** {info['cause']}")
                st.markdown(f"**Symptoms:** {info['symptoms']}")
                st.markdown("**Management Tips:**")
                for tip in info["management"]:
                    st.markdown(f"- {tip}")

            # New-image button
            if st.button("🔄 New Image"):
                st.experimental_rerun()
