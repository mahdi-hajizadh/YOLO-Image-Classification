import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
from ultralytics.utils import ASSETS


st.set_page_config(
    page_title="YOLO Image Classifier",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 YOLO Image Classifier")
st.caption("Image classification powered by Ultralytics YOLO26")


@st.cache_resource
def load_model():
    return YOLO("yolo26n-cls.pt")


model = load_model()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    photo = ImageOps.exif_transpose(
        Image.open(uploaded)
    ).convert("RGB")
else:
    photo = Image.open(
        ASSETS / "bus.jpg"
    ).convert("RGB")

if st.button("Classify Image", type="primary"):
    with st.spinner("Analyzing image..."):

        result = model(photo, verbose=False)[0]

        if result.probs is None:
            st.error("Classification failed.")
            st.stop()

        top5 = result.probs.top5
        confidences = result.probs.top5conf.tolist()

        predictions = [
            (result.names[int(index)], float(confidence))
            for index, confidence in zip(top5, confidences)
        ]

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(photo, use_container_width=True)

    with right:
        st.subheader("Classification Results")

        best_name, best_confidence = predictions[0]

        st.success(
            f"Prediction: {best_name}"
        )

        st.metric(
            "Confidence",
            f"{best_confidence * 100:.2f}%"
        )

        st.subheader("Top 5 Predictions")

        for name, confidence in predictions:
            st.progress(
                confidence,
                text=f"{name}: {confidence * 100:.2f}%"
            )
