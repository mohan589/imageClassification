import streamlit as st
import ssl
from transformers import pipeline
from PIL import Image

ssl._create_default_https_context = ssl._create_stdlib_context

classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    results = classifier(image)
    for result in results:
      print(f"Label: {result['label']}, Confidence: {result['score']:.2f}")
      st.write(f"Prediction: {result['label']}")
