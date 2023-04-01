import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, model_from_json
import tensorflow as tf
import cv2

st.set_page_config(page_title="Plant Disease Prediction", page_icon="🌴", layout="wide")

# Loading model
json_file = open(r'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"model.h5")
print("Loaded model from disk")
loaded_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer='adam', metrics=['accuracy'])

class_names = ['Pepper__bell___Bacterial_spot',
 'Pepper__bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy',]

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

with st.sidebar:
    st.title("✨ Webapp")

    with st.expander("Prediction", True):
        app = st.checkbox('App', value = True)
        analysis = st.checkbox('Analysis', value = True)

if app:
    st.title("Plant Disease Prediction")
    st.header("Upload Image")
    # file, link = st.columns(2)
    uploaded_file = st.file_uploader("Browse to upload image", type = ['png', 'jpg', 'jpeg'])
    
    if st.button(label = "Predict"):
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image)
            predicted_class, confidence = predict(loaded_model, image) 
            st.write(f"Predicted: {predicted_class}.")
            st.write(f"Confidence: {confidence}%")

if analysis:
    accuracy, output = st.columns(2)
    accuracy.title("Accuracy of Model")
    accuracy.image(r"https://github.com/Yadav-Roshan/Plant_Disease_Detection/blob/main/images/model_accuracy.png?raw=true")
    output.title("Validation Output")
    output.image(r"https://github.com/Yadav-Roshan/Plant_Disease_Detection/blob/main/images/output_labeled.png?raw=true")
