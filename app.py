import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/model.h5')

st.title("🧠 Neural Network Prediction App")
st.write("Enter input values:")

# Create dynamic inputs
input_data = []

num_features = model.input_shape[1]

for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)

    st.success(f"Prediction: {prediction[0][0]:.4f}")