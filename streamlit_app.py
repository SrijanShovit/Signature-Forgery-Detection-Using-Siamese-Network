import streamlit as st
import time
from PIL import Image
from src.ui_utils import (
    load_model,
    verify_signatures,
    generate_gradcam
)
from src.model_training_pipeline import TripletSiameseModel


st.set_page_config(page_title="Signature Verification", page_icon="✍️")

st.title("Signature Verification")


@st.cache_resource
def get_model():
    return load_model(TripletSiameseModel)

model = get_model()


col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader("Reference Signature", type=["png","jpg","jpeg"])

with col2:
    test_file = st.file_uploader("Test Signature", type=["png","jpg","jpeg"])


if ref_file and test_file:

    ref_img = Image.open(ref_file)
    test_img = Image.open(test_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(ref_img, caption="Reference", width="stretch")

    with col2:
        st.image(test_img, caption="Test", width="stretch")


    if st.button("Verify"):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        result = verify_signatures(model, ref_img, test_img)

        distance = result["distance"]
        same = result["same_signature"]

        st.subheader("Result")

        st.write(f"**Embedding Distance:** {distance:.4f} | **Threshold:** {result['threshold']:.4f}")
        if same:
            st.success("✔ Signatures MATCH")
        else:
            st.error("❌ Signatures DO NOT MATCH")


        # -------------------
        # GradCAM
        # -------------------

        st.subheader("Grad-CAM Visualization")

        cam1 = generate_gradcam(model, result["img1_tensor"])
        cam2 = generate_gradcam(model, result["img2_tensor"])

        col1, col2 = st.columns(2)

        with col1:
            st.image(cam1, caption="Reference Attention", width="stretch")

        with col2:
            st.image(cam2, caption="Test Attention", width="stretch")