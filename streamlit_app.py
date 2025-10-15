import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

IMG_SIZE = (150, 150)
USE_PREPROCESS_INPUT = False


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("intel_resnet50_colab.h5", compile=False)

@st.cache_data
def load_labels():
    with open("labels.json", "r") as f:
        class_indices = json.load(f)   
    inv = {v: k for k, v in class_indices.items()}
    class_names = [inv[i] for i in range(len(inv))]
    return class_names

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if USE_PREPROCESS_INPUT:
        pass
    else:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.set_page_config(page_title="Intel Scenes Classifier", page_icon="🌲", layout="centered")
st.title("Intel Natural Scenes Classifier")
st.caption("Upload an image to get the predicted scene and confidence.")

model = load_model()
class_names = load_labels()

uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    x = preprocess(img)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_label = class_names[top_idx]
    top_conf = float(probs[top_idx])

    st.subheader(f"This looks like a: {top_label.capitalize()}")
    st.write(f"Confidence: {top_conf*100:.2f}%")

else:
    st.info("Choose an image file to get a prediction.")