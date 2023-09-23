import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model and scaler
with open('gb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define the features
features = [
    "ทะเล Beach",
    "อยู่บ้าน Stay home",
    "ฟังเพลง Music",
    "ดูหนัง Movie",
    "เล่นเกมส์ Video Games",
    "เสต็ก Steak",
    "ส้มตำ Papaya salad Somtum",
    "สลัด Salad",
    "ซีรี่เกาหลี Korean dramas",
    "หนังซุปเปอร์ฮีโร่ Super heros",
    "สีฟ้า Blue",
    "สีขาว White",
    "ผมสั้น Short hair",
    "ชอบปาร์ตี้ Party",
    "ชอบดูดวง Horoscope"
]

# Create a dictionary to store the input values
input_values = {}

# Create UI elements to collect input
st.title("Mountain Liking Predictor")
for feature in features:
    input_values[feature] = st.slider(f"Rate your preference for {feature} (1-5):", 1, 5)

# Button to get the prediction result
if st.button("Predict"):
    # Convert input values to a numpy array and scale them
    input_data = np.array(list(input_values.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    
    # Display the prediction result
    if prediction == 1:
        st.success("You likely like mountains!")
    else:
        st.warning("You likely do not like mountains!")

