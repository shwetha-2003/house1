import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.header("Housing Price Prediction")

Algo = st.selectbox("Select Model", ('randomforest','xgboost', "catboost"))

# Dropdown bar 1
Borough = st.selectbox("Select Borough",(2,3,4) )

# Input bar 2
Longitude = st.number_input("Longitude", step=1e-8, format="%.7f")


# Input bar 3
Latitude = st.number_input("latitude", step=1e-8, format="%.7f")

# Input bar 4
Neighborhood = st.text_input("Neighborhood")

# Input bar 5
Residential_Units = st.number_input("Residential_Units")

# Input bar 6
Tax_Class_At_Time_Of_Sale = st.number_input("Tax_Class")

#Input bar 7
Building_Class_At_Time_Of_Sale =st.text_input("Building_Class")

#Input bar 8
age = st.number_input("age")

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    modelM = joblib.load(f"{Algo}.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[Borough, Neighborhood, Residential_Units, Tax_Class_At_Time_Of_Sale, Building_Class_At_Time_Of_Sale, Latitude, Longitude, age]], 
                     columns = ["Borough", "Neighborhood", "Residential_Units", "Tax_Class_At_Time_Of_Sale", "Building_Class_At_Time_Of_Sale", "latitude", "longitude", "age"])
    
    # Get prediction
    prediction = modelM.predict(X)
    avg = (np.exp(prediction)).round(2)[0]
    # Output prediction
    st.text(f"The Price prediction is {avg}")

import base64
file_ = open("kramer_gif.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
        f'<center><img src="data:image/gif;base64,{data_url}" alt="cat gif"></center>',
        unsafe_allow_html=True,
    )

