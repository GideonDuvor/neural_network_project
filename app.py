import streamlit as st
import pandas as pd
import pickle
import os

st.title("🧠 Neural Network Prediction App")

# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


model = load_model()

st.success("✅ Model loaded successfully")
st.write("Enter input values:")

# -----------------------------
# FIX: set feature count manually
# -----------------------------
num_features = 8

input_data = []

for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    st.success(f"Prediction: {prediction[0]}")
