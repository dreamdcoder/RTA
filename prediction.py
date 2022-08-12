# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import joblib
import streamlit as st


def preprocessing(df):
    # df.columns = df.columns.str.lower()

    df['Time'] = pd.to_datetime(df['Time'])
    df['hour'] = df['Time'].dt.hour
    df['minute'] = df['Time'].dt.minute
    df.drop("Time", inplace=True, axis=1)
    df.columns = df.columns.str.lower()
    # st.write(df.columns)
    df.drop(columns=['defect_of_vehicle', 'vehicle_driver_relation', 'work_of_casuality', 'fitness_of_casuality',
                     'service_year_of_vehicle'], inplace=True)
    missing_values_df= df.isnull().sum()
    # missing_values_df = df.isnull().sum().iloc[:, 1] != 0
    missing_values_df= pd.DataFrame(missing_values_df[missing_values_df!=0])
    # st.write("x")
    # st.write(missing_values_df)
    mode_df=pd.read_csv('D:\RTA\Model\mode.csv')
    mode_df.set_index('col',inplace=True)
    # st.write(mode_df)

    for col in missing_values_df.index.tolist():
        mode = mode_df.loc[col][0]
        # st.write(mode)
        df[col].fillna(mode, inplace=True)
    # st.write(df)
    enc = joblib.load("D:\RTA\Model\X_Encoder.pkl")
    X = enc.transform(df)
    return X

def prediction(df):
    model = joblib.load("D:\RTA\Model\model.pkl")
    y_enc=joblib.load(("D:\RTA\Model\Y_Encoder.pkl"))
    y=model.predict(df)
    y_val=y_enc.inverse_transform(np.array(y).reshape(-1,1))
    st.write(f" The person met with accident will have {y_val[-1,0]}.")

