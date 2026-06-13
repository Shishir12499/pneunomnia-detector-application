import json
from pathlib import Path

import h5py
import numpy as np
import streamlit as st
from PIL import Image

MODEL_PATH = Path("pneumonia_model.h5")
CLASS_NAMES_PATH = Path("class_names.json")


st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon=":hospital:",
    layout="centered",
)

try:
    import tensorflow as tf
except ModuleNotFoundError:
    st.error(
        "TensorFlow is not installed for this Python environment. "
        "Run: python -m pip install -r requirements.txt"
    )
    st.stop()


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        st.error("Missing class names file: class_names.json")
        st.stop()

    try:
        with CLASS_NAMES_PATH.open("r") as f:
            names = json.load(f)
    except json.JSONDecodeError:
        st.error("class_names.json is invalid. Use: [\"NORMAL\", \"PNEUMONIA\"]")
        st.stop()

    if not isinstance(names, list) or len(names) < 2:
        st.error("class_names.json must contain at least two labels.")
        st.stop()

    return names


def ensure_model_file():
    if MODEL_PATH.exists():
        return

    st.info(
        "Upload your trained Google Colab model file to continue. "
        "It will be saved as pneumonia_model.h5 in this project."
    )
    uploaded_model = st.file_uploader(
        "Upload trained model",
        type=["h5", "keras"],
        key="model_uploader",
    )

    if uploaded_model is None:
        st.stop()

    MODEL_PATH.write_bytes(uploaded_model.getbuffer())
    load_model.clear()
    st.success("Model uploaded successfully. Loading app...")
    st.rerun()


def remove_unsupported_config_keys(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for item in value.values():
            remove_unsupported_config_keys(item)
    elif isinstance(value, list):
        for item in value:
            remove_unsupported_config_keys(item)


def repair_h5_model_config(model_path):
    with h5py.File(model_path, "r+") as model_file:
        raw_config = model_file.attrs.get("model_config")

        if raw_config is None:
            return False

        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

        config = json.loads(raw_config)
        before = json.dumps(config, sort_keys=True)
        remove_unsupported_config_keys(config)
        after = json.dumps(config, sort_keys=True)

        if before == after:
            return False

        model_file.attrs.modify("model_config", json.dumps(config).encode("utf-8"))
        return True


@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        repaired = repair_h5_model_config(MODEL_PATH)
        if not repaired:
            raise

        return tf.keras.models.load_model(MODEL_PATH, compile=False)


st.title("Pneumonia Detection System")
st.write("Upload a Chest X-Ray image to predict Pneumonia.")

class_names = load_class_names()
ensure_model_file()
model = load_model()

uploaded_file = st.file_uploader(
    "Choose an X-Ray Image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded X-Ray",
        use_container_width=True,
    )

    img = image.resize((150, 150))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Predicting..."):
        try:
            prediction = model.predict(img_array, verbose=0)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

        probability = float(prediction[0][0])

        if probability > 0.5:
            result = class_names[1]
            confidence = probability * 100

            st.error(
                f"Prediction: {result}\n\nConfidence: {confidence:.2f}%"
            )

        else:
            result = class_names[0]
            confidence = (1 - probability) * 100

            st.success(
                f"Prediction: {result}\n\nConfidence: {confidence:.2f}%"
            )

st.markdown("---")
st.write("Developed using Streamlit and Deep Learning")
