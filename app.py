import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os


st.title('Obesity Level Prediction Application')

st.write('This is a simple application that can be used in the medical space to predict obesity level based on some features')

#Input features
Gender = st.selectbox('Gender',['male','Female'])
Age = st.slider('Age',value=20,min_value=0,max_value=100,step=1)
Weight = st.number_input('Weight',value=70.0,min_value=0.0,max_value=200.0,step=0.01)
family_history_with_overweight = st.selectbox('Family history with overweight',['yes','no'])
FAVC = st.selectbox('Frequent consumption of high caloric food',['yes','no'])
FCVC = st.number_input('FCVC',value=2.0,min_value=1.0,max_value=4.0,step=0.1)
SCC = st.selectbox('SCC',['yes','no'])
FAF = st.number_input('FAF',value=2.0,min_value=0.0,max_value=3.0,step=0.1)
CALC = st.selectbox('CALC',['no', 'Sometimes', 'Frequently', 'Always'])
MTRANS = st.selectbox('MTRANS',['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile'])


##loading serialized data
mo = pickle.load(open('model.pkl','rb'))
enc = pickle.load(open('encoder.pkl','rb'))
sca = pickle.load(open('scaler.pkl','rb'))

##create a Dataframe of features
input_data = pd.DataFrame([[Gender, Age, Weight, family_history_with_overweight, FAVC, FCVC, SCC, FAF, CALC, MTRANS]],
             columns=['Gender', 'Age', 'Weight', 'family_history_with_overweight', 'FAVC',
       'FCVC', 'SCC', 'FAF', 'CALC','MTRANS'])

input_data_encoded = enc.transform(input_data)
input_data_scaled = sca.transform(input_data_encoded)


##creating logic for prediction application
if st.button('Predict Obesity Level'):
    with st.spinner('Analyzing Data...'):
        prediction = mo.predict(input_data_scaled)  # Predict obesity level (0, 1, 2, 3)
        if prediction == 0:
            st.success('The predicted obesity level is: Normal Weight')
        elif prediction == 1:
            st.success('The predicted obesity level is: Obesity Type III')
        elif prediction == 2:
            st.success('The predicted obesity level is: Obesity Type I')
        elif prediction == 3:
            st.success('The predicted obesity level is: Insufficient Weight')
        elif prediction == 4:
            st.success('The predicted obesity level is: Obesity Type II')
        elif prediction == 5:
            st.success('The predicted obesity level is: Overweight Level II')
        else:
            st.success('The predicted obesity level is: Overweight Level I')
    st.balloons()