import streamlit as st
from silos.classification.prediction.prediction import SilosClassification
from silos.segmentation.prediction.prediction import SilosSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


ss = SilosSegmentation()
sc = SilosClassification()

j = 0
st.markdown(f"<h1 style='text-align: center;'>Foodix</h1>", unsafe_allow_html=True)
# st.title("Foodix")
st.header("Foodix detects silos on satellite pictures")
img_file_buffer = st.file_uploader(
    "Upload your satellite pictures",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

if img_file_buffer is not None:
    string = "Picture "
    tabs_names = ["Picture 1"]
    tabs_names_2 = [string + str(i + 2) for i in range(len(img_file_buffer) - 1)]
    tabs_names.extend(tabs_names_2)
    tabs = st.tabs(tabs_names)
    for uploaded_file_buffer in img_file_buffer:
        tab = tabs[j]
        with tab:
            prediction = sc.predict(uploaded_file_buffer, from_buffer=True)
            prediction = round(prediction, 2)
            st.markdown(
                f"<h1 style='text-align: center;'>Prediction</h1>",
                unsafe_allow_html=True,
            )
            st.write(
                "Foodix Classification model returns the probability of a silo being present on the uploaded "
                "satellite picture. Prediction range is 0 to 1 (the closer the prediction is to 1, the higher "
                "the chances of a silo)."
            )
            st.markdown("The prediction given by Foodix Model is:")
            st.markdown(
                f"<h1 style='text-align: center;'>" + str(prediction) + "</h1>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<h1 style='text-align: center;'>Segmentation</h1>",
                unsafe_allow_html=True,
            )
            st.write(
                "Foodix Segmentation model tries to locate potential silos on the uploaded satellite picture."
            )
            st.header("Uploaded Picture " + str(j + 1))
            st.write("Filename:", uploaded_file_buffer.name)

            predimask = ss.predict(uploaded_file_buffer, from_buffer=True)
            predimask = plt.imshow(tf.squeeze(predimask))
            predimask = Image.fromarray(
                np.uint8(predimask.get_cmap()(predimask.get_array()) * 255)
            )
            col1, col2 = st.columns(2)
            col1.image(uploaded_file_buffer)
            col2.image(predimask)
            j += 1
