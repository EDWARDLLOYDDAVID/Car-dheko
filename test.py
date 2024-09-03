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
city_dict = {'Bangalore' : 0, 'Chennai' :1 , 'Delhi': 2, 'Hyderabad':3, 'Jaipur':4, 'Kolkata':5}

st.sidebar.title('Input Fields')

btype = st.sidebar.selectbox('Body Type', options= bt,index = 0)
ownerNo = st.sidebar.number_input('Owner No', min_value=0, max_value=4)
modelYear = st.sidebar.number_input('Model Year', min_value=2015, max_value=2022)
Fuel_Type = st.sidebar.selectbox('Fuel Type',options= Fuel,index = 0)

Insurance_Velocity = st.sidebar.selectbox('Insurance Validity', options= Insurance,index = 0)
Kms_Driven = st.sidebar.number_input('Kms Driven', min_value=10000, max_value=150000)

Transmission = st.sidebar.selectbox('Transmission', options= Transmission,index = 0)
Year_of_Manufacture = st.sidebar.number_input('Year of Manufacture', min_value=2015, max_value=2022)
Mileage = st.sidebar.number_input('Mileage', min_value=10, max_value=30)
Gear_Box = st.sidebar.number_input('Gear Box', min_value=4, max_value=5)
City = st.sidebar.selectbox('City', options= City,index = 0)


def load_model():
    return pickle.load(open("best_model.pkl", "rb"))

model = load_model()

def preprocess_input(btype, ownerNo, modelYear, Fuel_Type, Insurance_Velocity, Kms_Driven, Transmission, Year_of_Manufacture, Mileage, Gear_Box, City):
    # Convert categorical variables to numerical values using the dictionaries
    bt_num = bt_dic[btype]
    fuel_num = fuel_dic[Fuel_Type]
    insurance_num = insurance_dic[Insurance_Velocity]
    transmission_num = transmission_dic[Transmission]
    city_num = city_dict[City]  # Corrected line

    # Create a numpy array with the preprocessed input data
    input_data = np.array([bt_num, ownerNo, modelYear, fuel_num, insurance_num, Kms_Driven, transmission_num, Year_of_Manufacture, Mileage, Gear_Box, city_num])

    # Reshape the array to match the model's input shape
    input_data = input_data.reshape(1, -1)

    return input_data

def make_prediction(input_data):
    # Use the loaded model to make a prediction
    prediction = model.predict(input_data)
    return prediction

# Create a button to trigger the prediction
if st.sidebar.button('Predict Price'):
    input_data = preprocess_input(btype, ownerNo, modelYear, Fuel_Type, Insurance_Velocity, Kms_Driven, Transmission, Year_of_Manufacture, Mileage, Gear_Box, City)
    prediction = make_prediction(input_data)
    st.write(f'Predicted Price: {prediction[0]:.2f}')  # Note the [0] indexing to get the first element of the prediction array