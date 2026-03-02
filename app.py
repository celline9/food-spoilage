import streamlit as st
import joblib
import numpy as np
from pathlib import Path

st.title("Food Spoilage Prediction")

BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "naive_bayes.pkl"

loaded_nb_model = joblib.load(model_path)

st.success("Naive Bayes model loaded successfully!")
st.write("Model type:", type(loaded_nb_model))
st.write("Model parameters:", loaded_nb_model.get_params())

fridge_days = st.number_input("Days in fridge", 0, 30)
freeze_months = st.number_input("Months in freezer", 0, 12)

if st.button("Predict"):
    X = np.array([[fridge_days, freeze_months]])
    pred = loaded_nb_model.predict(X)

    st.subheader("Prediction result")
    st.write(pred[0])
