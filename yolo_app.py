import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
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


uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if st.button("Classify Image", type="primary"):
    source = uploaded if uploaded is not None else ASSETS / "bus.jpg"
    try:
        with Image.open(source) as image:
            if image.format not in {"JPEG", "PNG"} or image.width * image.height > 25_000_000:
                st.error("Use a JPEG or PNG image with at most 25 megapixels.")
                st.stop()
            photo = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("Could not read the image. Try a valid JPEG or PNG file.")
        st.stop()

    with st.spinner("Analyzing image..."):
        try:
            result = load_model()(photo, verbose=False)[0]
        except (OSError, RuntimeError, ValueError):
            st.error("Classification could not run. Check that the model and its weights are available.")
            st.stop()

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
        st.subheader("Input image")
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
