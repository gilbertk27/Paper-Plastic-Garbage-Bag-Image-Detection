import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

def app():
    # title
    st.title('*Plastic, Paper, or Garbage Bag? - Computer Vision Project using Neural Network Algorithm*')

    # subheader
    st.subheader('EDA for Plastic, Paper, or Garbage Bag Classification')

    # add image
    image = Image.open('trash.jpg')
    st.image(image, caption = 'Plastic Bag')

    # Markdown
    st.markdown('----')

    # show dataframe
    df = pd.read_csv('image_df.csv')
    st.dataframe(df)
        
    # writing dataset explanation
    st.write('#### Dataset Explanation')
    st.write('''
    - **Filepath**: Represent the file path of the image.
    - **Label**: Represent the class of the image. There are 3 classes in this dataset: `paper`, `plastic`, and `garbage bag`.
    ''')

    st.write('#### Check for Image Size Scatterplot')
    st.image('scatterplot.png', caption='Image Size Scatterplot')
    st.write('''
    - From the scatter plot above, we can see that all the images have the same size of 300x300 pixels. 
    This is good because we don't have to resize the images before feeding them to the model. 
    If the images have different sizes, we have to resize them to the same size before feeding them to the model.
    ''')
        
    st.write('#### Check for Each Category Count')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(df['Label'])
    plt.title('Label Count')
    plt.xlabel('Label')
    plt.ylabel('Count')
    st.pyplot(fig)  
    st.write('''
    - After checking the image sizes, we can check the class balance.
    We can see that the classes are balanced. This is good because we don't have to do any class balancing. 
    In case of imbalanced classes, we can use data augmentation to balance the classes.
     ''')

    st.write('#### Check for Image Characteristics')
    st.image('characteristics.JPG', caption='Image Characteristics')
    st.write('''
    From the random image above for each class, we can see some of the characteristics of the images for each class:
    - Paper: The images are mostly brown colored and shaped rectangular. The position of the images are very random. Some of the images are rotated and some are not. Some of the images are also flipped horizontally and vertically. In this case we can use data augmentation to flip the images horizontally and vertically so that the model can learn the images better.
    - Plastic: The images are mostly very colorful (blue, green, white, etc) and shaped also rectangular but have little handle to grab. The position of the images are very random. Some of the images are rotated and some are not. Some of the images are also flipped horizontally and vertically. Similar to the paper images, we can use data augmentation to flip the images horizontally and vertically so that the model can learn the images better.
    - Garbage Bag: The images are mostly black colored and shaped rounder compared to the other classes. The position of the images are very random. Some of the images are rotated and some are not. Some of the images are also flipped horizontally and vertically. Again, we can use data augmentation to flip the images horizontally and vertically so that the model can learn the images better.
    ''')

    st.write('#### Additional Check for Image Colorspace')
    st.image('colorspace_red.png', caption='Image Colorspace Red')
    st.image('colorspace_green.png', caption='Image Colorspace Green')
    st.image('colorspace_blue.png', caption='Image Colorspace Blue')
    st.write('''
    - Since the images managed to show the color correctly according to the specified channel, we can conclude that the images colorspace is indeed RGB colospace images
    ''')


if __name__ == '__main__':
    app()