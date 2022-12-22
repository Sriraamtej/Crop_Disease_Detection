import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import numpy as np
import sys
from PIL import Image,ImageOps
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image

def load_lottiefile(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
lottie_insert=load_lottiefile(r"C:\Users\srira\Downloads\14482-welcome-onboard.json")
lottie_insert2=load_lottiefile(r"C:\Users\srira\Downloads\96262-detective-search.json")

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model(r"C:\Users\srira\Downloads\plant_disease_detection.h5")
    return model


model=load_model()

def import_and_predict(image_data,model):
    size=(224,224)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction




def main():
    st.title("Crop Disease Detection")
    html_temp="""
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Crop Disease Detector </h2>
    </div>
    """
    
    
    with st.sidebar:
        selected=option_menu(
        menu_title="Main Menu",
        options=["About","Crop Disease Detector"])
    if selected=="Crop Disease Detector":
        st.markdown(html_temp,unsafe_allow_html=True)
        file=st.file_uploader("please upload your image",type=["jpg","png","jpeg"])
        def import_and_predict(image_data,model):
            size=(224,224)
            image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
            img=np.asarray(image)
            img_reshape=img[np.newaxis,...]
            prediction=model.predict(img_reshape)
            return prediction

        if file is None:
            st.text("please upload an image file")
        else:
            image=Image.open(file)
            st.image(image,use_column_width=True)
            st_lottie(
            lottie_insert2,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            )
            predictions=import_and_predict(image,model)
            class_names=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            string="This image most likely is "+class_names[np.argmax(predictions)]
            st.success(string)
    if selected=="About":
        st_lottie(
            lottie_insert,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            )
        st.header('Application')
        st.write("Crop diseases are an important problem, as they cause serious reduction in quantity as well as quality of agriculture products. An automatic plant-disease detection system provides clear benefit in monitoring of large fields, as this is the only approach that provides a chance to discover diseases at an early stage")
        st.header('What is this page??')
        st.write("This page contains a predictive model that could assist the farmers in finding the disease that's affecting their crops ")
        st.write("On the following page, you can use the Crop Disease Detector to identify the disease of the plant")
        st.markdown("This page is created by Sriraam Tej")

if __name__=="__main__":
    main()

