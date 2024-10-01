import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

from PIL import Image

def main():
    st.title('Image Classifier')
    st.write('Upload any image that you think fits into one of the following categories and see if the prediction is correct:')
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        * airplane
        * automobile
        * bird
        * cat
        * deer
        """)

    with col2:
        st.markdown("""
        * dog
        * frog
        * horse
        * ship
        * truck
        """)

    file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
    if file:
        image = Image.open(file)

        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1, 32, 32, 3))

        model = tf.keras.models.load_model('models/imgClassifier_model.h5')

        predictions = model.predict(img_array)
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        top_prediction_idx = np.argmax(predictions[0])
        top_prediction_class = categories[top_prediction_idx]

        st.write(f"This image depicts a(n): **{top_prediction_class}**")
        st.image(image, use_column_width=True)


        fig, ax = plt.subplots()
        y_pos = np.arange(len(categories))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.invert_yaxis()
        ax.set_xlabel('Probability')
        ax.set_title('Image Predictions')

        st.pyplot(fig)
    else:
        st.text('You have not uploaded an image yet.')

if __name__ ==  '__main__':
    main()