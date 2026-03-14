import streamlit as st
from PIL import Image
import time

from src.ui_utils import load_model, verify_signatures
from src.model_training_pipeline import TripletSiameseModel


# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(
    page_title="Signature Verification",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Signature Verification")
st.write("Upload a reference signature and a test signature to verify.")


# --------------------------------
# LOAD MODEL
# --------------------------------

@st.cache_resource
def load_verification_model():
    return load_model(TripletSiameseModel)

model = load_verification_model()


# --------------------------------
# FILE UPLOAD
# --------------------------------

col1, col2 = st.columns(2)

with col1:
    ref_file = st.file_uploader(
        "Upload Reference Signature",
        type=["png", "jpg", "jpeg"]
    )

with col2:
    test_file = st.file_uploader(
        "Upload Test Signature",
        type=["png", "jpg", "jpeg"]
    )


# --------------------------------
# SHOW IMAGES
# --------------------------------

if ref_file and test_file:

    ref_img = Image.open(ref_file)
    test_img = Image.open(test_file)

    st.subheader("Uploaded Signatures")

    col1, col2 = st.columns(2)

    with col1:
        st.image(ref_img, caption="Reference", width="stretch")

    with col2:
        st.image(ref_img, caption="Test", width="stretch")


    # --------------------------------
    # VERIFY BUTTON
    # --------------------------------

    if st.button("Verify Signatures"):

        progress = st.progress(0)

        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        result = verify_signatures(model, ref_img, test_img)

        distance = result["distance"]
        threshold = result["threshold"]
        same = result["same_signature"]

        st.subheader("Verification Result")

        st.write(f"**Embedding Distance:** {distance:.4f}")
        st.write(f"**Threshold:** {threshold}")

        if same:
            st.success("✔ Signatures MATCH")
        else:
            st.error("❌ Signatures DO NOT MATCH")