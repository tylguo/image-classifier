import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from PIL import Image

def main():
    st.title('Cifar10 Web Classifier')
    st.write('Upload any image that you think fits into one of these classes and see if the prediction is correct.')

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)
    else:
        st.text('You have not uploaded an image yet.')

