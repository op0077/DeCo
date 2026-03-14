import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(".."))

from pipeline import deblurring
from PIL import Image   
import io
import numpy as np
import cv2
import base64

# ---------------------------
# Page Config (wide layout)
# ---------------------------
st.set_page_config(layout="wide")

# ---------------------------
# Background
# ---------------------------
def set_bg():
    with open("./assests/background_images/deco_bg_4.png", "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    css = f"""
    <style>

    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    h1 {{
        text-align:center;
        color:white;
        font-size:36px;
        font-weight:700;
        text-shadow:2px 2px 10px black;
        margin-bottom:5px;
    }}

    h3 {{
        margin-bottom:6px;
    }}

    .block-container {{
        padding-top:1rem;
        padding-bottom:0rem;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
    }}

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg()

# ---------------------------
# Title
# ---------------------------
st.markdown("<h1>✨ DeCo - AI Image Restoration</h1>", unsafe_allow_html=True)

st.markdown(
"""
<center style='color:white;font-size:14px'>
Deblur and Colorize images using AI models
</center>
""",
unsafe_allow_html=True
)

# ---------------------------
# Sidebar Controls
# ---------------------------
st.sidebar.title("⚙️ Controls")

task = st.sidebar.selectbox(
    "Select Operation",
    ["Deblurring", "Colorization", "Both"]
)

# Model options
if task == "Deblurring":
    model = st.sidebar.selectbox(
        "Select Deblurring Model",
        ["deblur_model1", "deblur_model2", "deblur_model3"]
    )

elif task == "Colorization":
    model = st.sidebar.selectbox(
        "Select Colorization Model",
        ["color_model1", "color_model2", "color_model3"]
    )

else:
    model_deblur = st.sidebar.selectbox(
        "Deblur Model",
        ["deblur_model1", "deblur_model2", "deblur_model3"]
    )

    model_color = st.sidebar.selectbox(
        "Colorization Model",
        ["color_model1", "color_model2", "color_model3"]
    )

# Upload image
image_file = st.sidebar.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

process = st.sidebar.button("🚀 Process Image")

# ---------------------------
# Main Area
# ---------------------------
if image_file is not None:

    col1, col2 = st.columns(2)

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    with col1:
        st.subheader("Original Image")
        st.image(image, width=350)

    if process:

        result = image.copy()

        # Deblurring
        if task in ["Deblurring", "Both"]:
            func_deblur = deblurring.Deblurr(result)
            result = func_deblur.apply_deblur()

        # Dummy colorization
        if task in ["Colorization", "Both"]:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        result_img = Image.fromarray(result)

        with col2:
            st.subheader("Processed Image")
            st.image(result_img, width=350)

        # Download Button (below images)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.markdown("<br>", unsafe_allow_html=True)

        center_col = st.columns([1,1,1])[1]

        with center_col:
            st.download_button(
                label="⬇ Download Image",
                data=byte_im,
                file_name="generated_image.png",
                mime="image/png"
            )