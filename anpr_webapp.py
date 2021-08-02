import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


import lp_detection
import lp_char_seg
import lp_char_recog
import lp_tesseract
from PIL import Image

st.set_page_config(page_title='License Plate Recognition', page_icon='🚗')

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)



padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.title('License Plate Recognition Model Test')

dmax = st.sidebar.slider('Dmax value', 288, 700, 608, step=30, key='dmax')
dmin = st.sidebar.slider('Dmin value', 128, 608, 288, step=30, key='dmin')
digit_w = st.sidebar.slider('Digit width for characters', 10, 60, 30, key='digit_w')
digit_h = st.sidebar.slider('Digit height for characters', 10, 150, 80, key='digit_h')
ratio_up = st.sidebar.selectbox('Upper limit for character bounding rectangle ratio', 
                                options=np.arange(2, 12), index=8, key='ratio_up')
hp = st.sidebar.slider('Bounding box height by plate height', 0.22, 0.79, 0.5, key='hp')


uploaded_img = st.file_uploader('Upload an image',
                               type='jpg', 
                               help='an image file with .jpg extension to be uploaded from your local FS')


if uploaded_img:
   
    st.image(uploaded_img, caption='license plate', width=500)
    test_image = Image.open(uploaded_img)
    test_image = np.array(test_image)

    if st.button('Get License Plate Number'):

        vehicle, LpImg, cor = lp_detection.get_license_plate(test_image, dmax, dmin)
     
        st.subheader('License Plate Detection')
        fig1, ax1 = plt.subplots()
        ax1.axis(False)
        ax1.imshow(LpImg[0])
        st.pyplot(fig1)


        plate, binary, dilated, blur, gray = lp_char_seg.plate_preprocessing(LpImg)
        cont = lp_char_seg.find_contours(binary)
        test_roi, crop_characters = lp_char_seg.store_chars(hp, digit_w, digit_h, ratio_up, plate, dilated, cont, binary)

        st.subheader('Character Segmentation')
        fig3, ax3 = plt.subplots()
        ax3.axis(False)
        ax3.imshow(test_roi)
        st.pyplot(fig3)


        plate_number = lp_char_recog.get_plate_number(crop_characters)
        st.subheader('License Plate Number - MobileNetV2 Result')
        st.header(plate_number)

        text = lp_tesseract.OCR(blur)
        st.subheader('License Plate Number - Pytesseract Result')
        st.header(text)

        st.write('Is the prediction wrong? Try tuning the parameters for a better result')

        st.balloons()
        

 



  
        
        

