import os
from io import StringIO

import streamlit as st
import pandas as pd
from prediction import preprocessing,prediction

uploaded_file = st.file_uploader("Choose a file")

global dataframe
global btn
df =None


def upload_file():
     #st.write("""# My first app Hello *world!*""")

     global btn
     if uploaded_file is None:
         btn = st.button('Predict', disabled=True, key=11)
     else:
         df = pd.read_csv(uploaded_file)
         btn = st.button('Predict', disabled=False, key=12)
         if btn:
             X = preprocessing(df)
             prediction(X)

if __name__ == '__main__':
     st.write(os.getcwd())
     upload_file()






