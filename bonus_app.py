import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import numpy as np
from PIL import Image, UnidentifiedImageError
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

def preprocess_one(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if USE_PREPROCESS_INPUT:
       
        pass
    else:
        arr = arr / 255.0
    return arr  

st.set_page_config(page_title="Intel Scenes Classifier", page_icon="🌲", layout="centered")
st.title("Intel Natural Scenes Classifier")
st.caption("Upload one or more images. The model will predict the scene and show its confidence.")

model = load_model()
class_names = load_labels()

uploaded_files = st.file_uploader(
    "Upload JPG/PNG images (you can select multiple files)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if uploaded_files:
  
    images_pil = []
    arrays = []
    names = []

    for f in uploaded_files:
        try:
            img = Image.open(f)
            arr = preprocess_one(img)
        except (UnidentifiedImageError, OSError):
            st.warning(f"Could not read file: {f.name}")
            continue
        images_pil.append(img)
        arrays.append(arr)
        names.append(f.name)

    if len(arrays) == 0:
        st.error("No valid images to process.")
    else:
        batch = np.stack(arrays, axis=0) 
        probs_batch = model.predict(batch, verbose=0)  

        for i, (img, probs, fname) in enumerate(zip(images_pil, probs_batch, names), start=1):
            top_idx = int(np.argmax(probs))
            top_label = class_names[top_idx]
            top_conf = float(probs[top_idx])

            c1, c2 = st.columns([2, 1], vertical_alignment="top")
            with c1:
                st.image(img, caption=f"{i}. {fname}", use_column_width=True)
            with c2:
                st.subheader(top_label.capitalize())
                st.write(f"Confidence: {top_conf*100:.2f}%")
        
else:
    st.info("Choose one or more image files to get predictions.")