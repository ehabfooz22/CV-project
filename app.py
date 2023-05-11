import keras
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import time
st.set_page_config(page_title="Brain Tumor Classifier", page_icon='images\istockphoto-1250205787-612x612.jpg')

st.title('Brain Tumor Classifier')

st.image('images\pexels-anna-shvets-4226219.jpg')
def classifier(img, model):

    model=keras.models.load_model(model, compile=False)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage,(128,128))
    img = img.reshape(1,128,128,3)
    prediction = model.predict(img)
    return prediction

def main():

    menu=['Home','Classifier','About']
    choice=st.sidebar.selectbox('Menu',menu)
    check=False

    if(choice=='Home'):

        st.subheader('What is a Brain Tumor\n')
        st.markdown("""\n 
            ##### A cancerous or non-cancerous mass or growth of abnormal cells in the brain.
           ##### Tumours can start in the brain, or cancer elsewhere in the body can spread to the b rain.
           ##### Symptoms include new or increasingly strong headaches, blurred vision, loss of balance, confusion and seizures. In some cases, there may be no symptoms.
           ##### Treatments include surgery, radiation and chemotherapy.
        """)

        st.subheader('Symptoms Of Brain Tumor\n')
        st.markdown("""
            \n\n
            1.Headache\n
            2.Muscle Weakness\n
            3.Nausea\n
            4.Vomitting\n
            5.Dizziness\n
            6.Inability To speak\n
            7.Mental Confusion\n
        """)

        st.subheader('TreatMent')
        st.markdown("""
            \n\n\
            1.**Chemotherapy**->Unwanted reactions to drugs given for the purpose of killing cancer cells.
            2.**Craniotomy**->Brain surgery in which a piece of bone is removed from the skull.
            3.**Tomotherapy**->Cancer treatment that aims high-dose radiation at tumours from many directions. Reduces damage to nearby tissue.    
            4.**Radiation therapy**->Treatment that uses x-rays and other high-energy rays to kill abnormal cells.
        """)

        st.write("**For More Information** [Click Here](https://www.cancer.net/cancer-types/brain-tumor/symptoms-and-signs)")

    elif(choice=='Classifier'):
        st.subheader('How To Use the App')
        st.write('Just Upload your **MRI scan** and the Classifier will tell you whether you have Tumor or not..')

        uploaded_file = st.file_uploader("Choose an imgae...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.spinner('Wait for it...')
            time.sleep(2)
            st.image(image, use_column_width=False)            
            st.success('Uploading Image Is Done!', icon="✅")

            st.write("Please Wait for your RESULT...")
            label = classifier(image, 'Models\model.h5')

            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.1)
                my_bar.progress(percent_complete + 1)

            if label < 0.5:
                st.success("**TEST RESULT: Not Tumor**")
                st.balloons()
            else:
                st.warning("**TEST RESULT: Tumor**", icon="⚠️")


    elif(choice=='About'):
        st.markdown("""
            The App is Build Purely in Python Programming Language.\n
            This App Uses Convolutional Neural Network for the Image Classification.\n
            Python's Deep Learning Library **Keras** was used in this App, which uses **Tensorflow** in the backend.\n
            The Dataset was Obtained From Kaggle->[Click Here](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)\n
        """)
        expander=st.expander('Group Members For the Project')
        expander.markdown("""
            **Eslam Ashraf**\n
            **Ehab Tarek**\n
            **Eid Abdul Rahim**\n
            **Ahmed Abdallah**\n
            **Ahmed Mohamed Kamel**\n
            **Engy Essam**\n
            **Assma Ahmed**\n

        """)

if __name__ == "__main__":
    main()