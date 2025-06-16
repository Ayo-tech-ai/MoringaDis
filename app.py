# --------------------------- imports ---------------------------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tf_explain.core.grad_cam import GradCAMPlusPlus as GradCAM

# ------------------------ disease info ------------------------
DISEASE_INFO = {
    "Bacterial Leaf Spot": {
        "name": "Bacterial Leaf Spot",
        "cause": "Bacteria (Xanthomonas / Pseudomonas spp.) infecting leaf tissue",
        "symptoms": (
            "Small, water-soaked specks that enlarge, darken and may ooze under humid conditions."
        ),
        "management": [
            "Remove and destroy infected leaves.",
            "Avoid overhead watering; keep foliage dry.",
            "Apply copper-based bactericide early in the outbreak.",
        ],
    },
    "Cercospora Leaf Spot": {
        "name": "Cercospora Leaf Spot",
        "cause": "Fungal infection by *Cercospora moringicola*",
        "symptoms": (
            "Circular grey-brown lesions with yellow halos; severe cases cause premature leaf drop."
        ),
        "management": [
            "Collect and burn fallen leaves to reduce spores.",
            "Apply protectant fungicide (e.g. mancozeb) during rainy periods.",
            "Prune overcrowded branches to improve air flow.",
        ],
    },
    "Yellow Leaf": {
        "name": "Nutrient / Water Stress (Yellow Leaf)",
        "cause": "Typically nitrogen or iron deficiency; sometimes over-watering",
        "symptoms": (
            "Uniform yellowing beginning on older leaves; veins may stay green if iron is lacking."
        ),
        "management": [
            "Apply balanced NPK fertiliser or iron chelate.",
            "Check soil drainage; avoid prolonged water-logging.",
            "Mulch to maintain even soil moisture.",
        ],
    },
}

CLASS_NAMES = list(DISEASE_INFO.keys())

# ------------------------ load model ------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("moringa_effnet_final.keras")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model

model = load_model()
backbone = model.get_layer("efficientnetb0")

# ---------------------- grad-cam helper ----------------------
def generate_gradcam(image_batch, model, class_idx, target_layer="block4c_project_conv"):
    explainer = GradCAM()
    cam = explainer.explain(
        validation_data=(image_batch, None),
        model=model,
        class_index=class_idx,
        layer_name=target_layer
    )
    return cam

# -------------------------- UI --------------------------
st.set_page_config(page_title="Moringa Leaf Disease Detector", layout="wide")
st.title("🌿 Moringa Leaf Disease Detector with Grad-CAM")

uploaded_file = st.file_uploader("Upload a moringa leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 1. Display image
    pil_img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    # 2. Preprocess
    arr = np.array(pil_img, dtype="float32")
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    # 3. Predict
    preds = backbone(arr, training=False)
    class_idx = int(np.argmax(preds))
    class_name = CLASS_NAMES[class_idx]
    st.success(f"🧠 Predicted class: **{class_name}**")

    # 4. Grad-CAM heatmap
    heatmap = generate_gradcam(arr, backbone, class_idx)
    st.image(heatmap, caption="Grad-CAM++ Heatmap", use_column_width=True)

    # 5. Disease info
    info = DISEASE_INFO[class_name]
    with st.expander("🩺 Disease Information", expanded=True):
        st.markdown(f"### {info['name']}")
        st.caption(f"**Cause:** {info['cause']}")
        st.markdown(f"**Symptoms:** {info['symptoms']}")
        st.markdown("**Management Tips:**")
        for tip in info["management"]:
            st.markdown(f"- {tip}")
