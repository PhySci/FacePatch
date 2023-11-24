import streamlit as st
from pipeline import pipeline
from PIL import Image

st.title('Test application')

uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"], on_change=None, args=None, kwargs=None)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img = Image.open(uploaded_file)

    img2 = pipeline(img)

    st.image(img2)

