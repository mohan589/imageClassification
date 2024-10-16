import streamlit as st
import torch
from PIL import Image

from transformers import pipeline

checkpoint = "openai/clip-vit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-image-classification")

# Load your model here
# model = torch.load('your_model.pth')
# model.eval()

def classify_image(image):
    # Preprocess and classify the image
    # return the predicted class
    pass

st.title("Image Classification App")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl", "jewel"])
    # prediction = classify_image(image)
    
    #only give results with score > 0.3, and it's label
    predictions = [p for p in predictions if p['score'] > 0.3]
    st.write(f"Prediction: {predictions}")
