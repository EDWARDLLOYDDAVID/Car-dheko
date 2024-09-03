import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Car Dheko Used Car Price Prediction")

bt = ['Convertibles', 'Coupe', 'Hatchback', 'Hybrids', 'MUV', 'Minivans', 'Pickup Trucks', 'SUV', 'Sedan', 'Wagon']
Fuel = ['CNG', 'Diesel', 'Electric', 'LPG', 'Petrol']
Insurance = ['Comprehensive', 'First Party insurance', 'Not Available', 'Second Party insurance', 'Third Party insurance', 'Zero Depreciation']
Transmission = ['Automatic', 'Manual']
City = ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Jaipur', 'Kolkata']

bt_dic = {'Convertibles' : 0, 'Coupe' : 1, 'Hatchback' : 2, 'Hybrids' : 3, 'MUV' : 4, 'Minivans' : 5, 'Pickup Trucks' : 6, 'SUV' : 7, 'Sedan' : 8, 'Wagon': 9}
fuel_dic = {'CNG' : 0, 'Diesel' : 1, 'Electric' : 2, 'LPG' : 3, 'Petrol' : 4}
insurance_dic = {'Comprehensive' : 0 , 'First Party insurance' : 1, 'Not Available' :2 , 'Second Party insurance':3, 'Third Party insurance':4, 'Zero Depreciation':5}
transmission_dic = {'Automatic' : 0, 'Manual' : 1}
city = {'Bangalore' : 0, 'Chennai' :1 , 'Delhi': 2, 'Hyderabad':3, 'Jaipur':4, 'Kolkata':5}

st.sidebar.title('Input Fields')

btype = st.sidebar.selectbox('Body Type', options= bt,index = 0,keys="bt")
ownerNo = st.sidebar.number_input('Owner No', min_value=0, max_value=4)
modelYear = st.sidebar.number_input('Model Year', min_value=2015, max_value=2022)
Fuel_Type = st.sidebar.selectbox('Fuel Type',options= Fuel,index = 0,keys="Fuel")

Insurance_Velocity = st.sidebar.selectbox('Insurance Validity', options= Insurance,index = 0,keys="Insurance")
Kms_Driven = st.sidebar.number_input('Kms Driven', min_value=10000, max_value=150000)

Transmission = st.sidebar.selectbox('Transmission', options= Transmission,index = 0,keys="Transmission")
Year_of_Manufacture = st.sidebar.number_input('Year of Manufacture', min_value=2015, max_value=2022)
Mileage = st.sidebar.number_input('Mileage', min_value=10, max_value=30)
Gear_Box = st.sidebar.number_input('Gear Box', min_value=4, max_value=5)
City = st.sidebar.selectbox('City', options= City,index = 0,keys="City")


def load_model():
    return pickle.load(open("best_model.pkl", "rb"))

model = load_model()


def predict_resale_price(user_inputs):
    try:
        # Combine user inputs to an array
        user_data = np.array([[btype, ownerNo, modelYear, Fuel_Type, Insurance_Velocity, Kms_Driven, Transmission, Year_of_Manufacture, Mileage, Gear_Box, City]])
        
        # Make prediction
        prediction = model.predict(user_data)
        
        return prediction[0]
    except Exception as e:
        return str(e)

# Create button to make prediction
with st.sidebar:
    pred_price_button = st.button("Estimate Used Car Price")

if pred_price_button:
    result = predict_resale_price({
        "btype": btype,
        "ownerNo": ownerNo,
        "modelYear": modelYear,
        "Fuel_Type": Fuel_Type,
        "Insurance_Velocity": Insurance_Velocity,
        "Kms_Driven": Kms_Driven,
        "Transmission": Transmission,
        "Year_of_Manufacture": Year_of_Manufacture,
        "Mileage": Mileage,
        "Gear_Box": Gear_Box,
        "City": City
    })
    st.write(f"The estimated used car price is: {result:.2f} 
