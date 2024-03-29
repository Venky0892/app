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
import urllib.request
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
    st.title("Objection Detection App")
    
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

    
    choose = st.sidebar.selectbox('Choose Model', ('yolo', 'fastrcnnresnt50fast', 'fastrcnnresnet50'))
    category = st.sidebar.selectbox('Choose the category', ('chair','bench','buffet','sofa', 'coffee-table','planter','side-table','lifestyle'))


    if st.sidebar.button("Process"):
        
        if 'fastrcnnresnt50' in choose:
            scoring_uri = "http://20.97.146.230:80/api/v1/service/gpu-cluster2/score" 
            #http://20.80.224.182:80/api/v1/service/automl-image-best-fastrcnn/score
            #http://20.97.146.230:80/api/v1/service/gpu-cluster2/score
                # date 27/01/2022 best so far 
            #key = 'EOrnj7U2vznn80ivScnw64het1hejtuw'   
            key = 'FBLyfUOpQe8PS8DuGKNxTO3z63fKcSUA'
            try:
                image_loading(scoring_uri, key, category)
            except ValueError:
                st.error("The model is in developement")

        if 'yolo' in choose:
            # scoring_uri = "http://20.80.224.182:80/api/v1/service/automl-image-7k-images/score" # 7k Images - Mean average precision Yolo
            # key = 'f85tEoGIHpXX6r57qfspDnXKKdtUpbA2'
            # scoring_uri = "http://20.97.146.230:80/api/v1/service/venkat35/score"
            # key = 'f85tEoGIHpXX6r57qfspDnXKKdtUpbA2'
            # scoring_uri = "http://20.80.224.182:80/api/v1/service/automl-image-best-hyper/score" # hyper-best
            # key = 'p3EYDvgcZ1meIlYjQvw7REDXgRr0RVEw' # hyper (yolo)

            scoring_uri = "http://20.97.146.230:80/api/v1/service/app-demo/score"
            key = 'd22z4wWv0F9cLrIeqGfR7xnv40CxvavU'

            image_loading(scoring_uri, key, category)

        if 'fastrcnnresnt50fast' in choose:
            scoring_uri = "http://20.80.224.182:80/api/v1/service/automl-image-best-fastrcnn-11k/score"
            # key = 'KzxANdalm3l87oakfI3eUi1apocJ51RT'
            try:
                image_loading(scoring_uri, key, category)
            except ValueError:
                st.write("The model is in developement")

def image_loading(scoring_uri, key, category):

    
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
    with st.spinner("Processing data..."):
        #########################################################
        for filename in os.listdir(images):

            # Load image data
            data = open(images + filename, 'rb').read()

            # Set the content type
            headers = {'Content-Type': 'application/octet-stream'}

            # If authentication is enabled, set the authorization header
            headers['Authorization'] = f'Bearer {key}'
            # headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ key)}
            # resp = urllib.request.Request(scoring_uri, data, headers)
            # Make the request and display the response
            resp = requests.post(scoring_uri, data, headers=headers)
            n_new, gtruth, pred = inf.load_image(images + filename, resp.text, n, category)
            name.append(filename)
            ground_truth.append(gtruth)
            predicted.append(pred)
        try:
            st.metric(label = 'Predicted no of class: ', value = inf.total_value(n_new))
        except AttributeError:
            st.write("Check for streamlit version which has metric method")
        # st.sidebar.write("Correctly predicted: ", inf.total_value(n))

   
        data = {'Filename': name,
            'Ground_Truth': ground_truth,
            'Predicted': predicted
            }   
        df = pd.DataFrame(data)
        df["Ground_Truth"].replace({category: 1}, inplace=True)
        df['Truth'] = df['Ground_Truth'].apply(lambda x: 0 if x == category else 1)
        df['New_predict'] = df['Predicted'].apply(lambda x: 1 if x == category else 0)
        csv = convert_df(df)
        colum = ["desk","dining-table","chair","dining-chair","sofa","sectional","coffee-table","side-table","shelving","planter","wall-mirror","basket","buffet","chandeliers","media-storage","dresser","ottoman","bench","office-chair","crib","bed"]
        try:
            dc = confusion_matrix(ground_truth, predicted, labels = colum)
            df_cm = pd.DataFrame(dc)
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            plt.figure(figsize = (20,20))
            # sn.set(font_scale=1.4)#for label size
            st.title("Heat Map")
            sn.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 15}, xticklabels = colum, yticklabels = colum, )
            st.pyplot(plt)
        except:
            pass



        # # st.pyplot()
        # acc = accuracy_score(df['Truth'], df['New_predict'])
        # st.sidebar.write("Accuracy", acc )
        # st.sidebar.write("Dataframe", df)
        # st.sidebar.download_button(
        #     label="Download data as CSV",
        #     data=csv,
        #     file_name='large_df.csv',
        #     mime='text/csv',
        # )
        # st.sidebar.write("Confusion_Matrix", dc)
        st.balloons()

def show():
    st.write(
        """
        ## 💯 Counter
        
        The most basic example: Store a count in `st.session_state` and increment when 
        clicked.
        """
    )
    if "counter" not in st.session_state:
        st.session_state.counter = 0

    def increment():
        st.session_state.counter += 1

    st.write("Counter:", st.session_state.counter)
    st.button("Plus one!", on_click=increment)

    if st.session_state.counter >= 50:
        st.success("King of counting there! Your trophy for reaching 50: 🏆")
    elif st.session_state.counter >= 10:
        st.warning("You made it to 10! Keep going to win a prize 🎈")

def feedback_box(up, down, check_false, pred, file_name_false, filename, check_true, file_name_true):

    if st.sidebar.checkbox(label = 'Right',  key = up):
        while True:
            st.write(":thumpsup:")
            check_true.append(pred)
            file_name_false.append(filename)

    elif st.sidebar.checkbox(label = 'Down',  key = down):
        while True:
            st.write(":thumpsup:")
            check_false.append(pred)
            file_name_true.append(filename)


        

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

    main()
