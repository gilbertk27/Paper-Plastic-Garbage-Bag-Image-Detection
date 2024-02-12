
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image


with open("model.pkl", "rb") as f: # load the model
    model = pickle.load(f)

with open('classes.txt', 'r') as file:
    class_names = [line.strip() for line in file]
    
def resize_image(image):
    scale_image = image.resize((300, 300))
    scaled_image = np.array(scale_image)
    st.image(scaled_image, caption='Processed Image')
    return scaled_image

def app():
    
    # write short description about the model
    st.write('''
    # *Plastic, Paper, or Garbage Bag? - Computer Vision Project using Neural Network Algorithm*
    - The model used for this detection is `Neural Network` Classifier model.
    - The model also used `Resnet50` to enhance the accuracy of the model during training.
    - This model managed to achieved `0.98` accuracy score on the test set to detect plastic, paper, or garbage bag.
    ''')
    
    # upload image
    input_file = st.file_uploader("Upload image here", type=[ "jpeg", "jpg", "png"])

    # display uploaded file
    if input_file is not None:
        image = Image.open(input_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Upload image first")
        

    # predict
    if st.button("Classify Image"):
        if input_file is not None:
            # resize images to 300x300
            processed_image = resize_image(image)
            x = tf.keras.utils.img_to_array(processed_image) # to change the image into an array
            input_image = np.expand_dims(x, axis=0) # Expand the shape of the array, for example 1D to 2D, 0 means rows/horizontal [[1, 2]]
            images = np.vstack([input_image])
        else:
            st.error("No image upload detected")
            
        # feed image to the model
        classes = model.predict(images)

        class_idx = np.argmax(classes, axis=-1)

        # use html to display the result and center the text
        st.write('<h1 style="text-align: center"> Predicted class:<br> {}'.format(class_names[class_idx[0]]), unsafe_allow_html=True)
        
if __name__ == '__main__':
  app()


