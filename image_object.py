from turtle import onclick
import streamlit as st
import pickle
from pathlib import Path
import pandas as pd 
import requests
from image_section import Inference
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import time
from typing import Dict
import numpy as np 
import seaborn as sn
# %matplotlib inline
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



inf = Inference()


def main():
    body()
    sidebar()

def body():
    st.title("Detection Feedback App")
    st.markdown(
        """
       This application is used to access the model performance by feedback mechanism.
        """, unsafe_allow_html=True)

# @st.cache
def sidebar():

    st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 1500px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
 
def thumpsup(up):
    thumpsup = st.sidebar.checkbox(label = 'Right',  key = up)
    return thumpsup

def thumpsdown(down):
    thumpsdown = st.sidebar.checkbox(label = 'Wrong',  key = down)

    return thumpsdown

def form_callback():
    st.write(st.session_state.my_slider)
    st.write(st.session_state.my_checkbox)

def image_loading():

    # choose = st.sidebar.selectbox('Choose Model', ('fastrcnnresnt50fast', 'yolo', 'fastrcnnresnet50'))
    category = st.sidebar.selectbox('Choose the category', ('chair','bench','buffet','sofa', 'coffee-table','planter','side-table'))

    images = "test/" + category + "/"
    count_images = len(os.listdir(images))
    # txt = st.text_area('Number of Images')
    

    st.sidebar.write('count_of_images: ', count_images)
    n = 0
    ground_truth = list()
    predicted = list()
    name = list()
    check_false = list()
    file_name_false = list()
    check_true = list()
    file_name_true = list()
    count = 100
    down = 200
    up = 300
    process = 400
    
    st.write(
        """  
        This is one for all machine learning fans: User Feeback from model predicted images and all of your 
        annotations are preserved in `st.session_state`!
        """
    )

    script_path = os.path.dirname(__file__)
    rel_path = "test"
    abs_file_path = script_path + "/" + rel_path + "/" + category
    files = os.listdir(abs_file_path)

    scoring_uri = "http://20.80.224.182:80/api/v1/service/yolo-9056p/score"
    key = 'f85tEoGIHpXX6r57qfspDnXKKdtUpbA2'


    if "annotations" not in st.session_state:
                st.session_state.annotations = {}
                st.session_state.files = files
                st.session_state.current_image = files[0]

    def annotate(label):
        st.session_state.annotations[st.session_state.current_image] = label
        if st.session_state.files:
            st.session_state.current_image = random.choice(st.session_state.files)
            
            st.session_state.files.remove(st.session_state.current_image)

#     
    image_path = (abs_file_path + "/" + st.session_state.current_image)

    col1, col2 = st.beta_columns(2)
    data = open(image_path, 'rb').read()

    # Set the content type
    headers = {'Content-Type': 'application/octet-stream'}

    # If authentication is enabled, set the authorization header
    headers['Authorization'] = f'Bearer {key}'

    # Make the request and display the response
    resp = requests.post(scoring_uri, data, headers=headers)
    n, gtruth, pred = inf.load_image(image_path, resp.text, n, category)
    name.append(image_path)
    ground_truth.append(gtruth)
    predicted.append(pred)
    # col1.image(image_path, width=300)
    with col2:
        if st.session_state.files:
            st.write(
                "Annotated:",
                len(st.session_state.annotations),
                "– Remaining:",
                len(st.session_state.files),
            )
            st.sidebar.button("This is a True!", on_click=annotate, args=("Correct",), key = str(up))
            st.sidebar.button("This is a False!", on_click=annotate, args=("Wrong",), key = str(down))
        else:
            st.success(
                f"🎈 Done! All {len(st.session_state.annotations)} images annotated."
            )
    st.write("### Annotations")
    # st.write(st.session_state.annotations)
    st.write(pd.DataFrame.from_dict(st.session_state.annotations, orient='index'))
    
    up += 1
    down += 2
   
    # st.balloons()
    
    

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def store():
    
    static_store = get_static_store()

    st.info(__doc__)

    result = st.sidebar.file_uploader("Upload")
    if result:
        # Process you file here
        value = result.getvalue()

        # And add it to the static_store if not already in
        if not value in static_store.values():
            static_store[result] = value
    else:
        static_store.clear()  # Hack to clear list if the user clears the cache and reloads the page
        st.sidebar.info("Upload one or more `.jpg` or 'jpeg' or 'png' files.")

    if st.button("Clear file list"):
        static_store.clear()
    if st.checkbox("Show file list?", True):
        st.write(list(static_store.keys()))
    if st.checkbox("Show content of files?"):
        for value in static_store.values():
            st.code(value)

    return result


if __name__=='__main__':

    image_loading()








