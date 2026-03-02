import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

st.title("Food Spoilage Prediction")
st.write("Predict whether your food is spoiled based on storage duration")

# Get base directory - use current working directory for Streamlit Cloud compatibility
try:
    BASE_DIR = Path(__file__).parent.resolve()
except:
    BASE_DIR = Path.cwd()

# Try multiple possible locations for files
def find_file(filename):
    possible_paths = [
        BASE_DIR / filename,
        Path.cwd() / filename,
        Path(filename)
    ]
    for path in possible_paths:
        if path.exists():
            return path
    # If not found, return the first option and let it fail with clear error
    return possible_paths[0]

model_path = "naive_bayes.pkl"
data_path = "food_spoilage.csv"

# Load model
try:
    loaded_nb_model = joblib.load(str(model_path))
except Exception as e:
    st.error(f"Error loading model from {model_path}: {e}")
    st.stop()

# Load food data
try:
    df = pd.read_csv(str(data_path), sep=';')
    df['max_fridge_days'] = pd.to_numeric(df['max_fridge_days'], errors='coerce')
    df['max_freezer_months'] = pd.to_numeric(df['max_freezer_months'], errors='coerce')
except Exception as e:
    st.error(f"Error loading data from {data_path}: {e}")
    st.error(f"Current directory: {Path.cwd()}")
    st.error(f"Base directory: {BASE_DIR}")
    st.stop()

# Get unique food items with their max values
food_info = df.groupby('food_item').agg({
    'max_fridge_days': 'first',
    'max_freezer_months': 'first',
    'category': 'first',
    'risk': 'first'
}).reset_index()

st.sidebar.header("Model Info")
st.sidebar.success("Naive Bayes Model Loaded")
st.sidebar.write(f"**Model Type:** {type(loaded_nb_model).__name__}")

# Main prediction interface
st.header("Select Food Item")
food_items = sorted(food_info['food_item'].unique())
selected_food = st.selectbox("Choose a food item:", food_items)

# Display food information
if selected_food:
    food_data = food_info[food_info['food_item'] == selected_food].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Category", food_data['category'])
    with col2:
        st.metric("Risk Level", food_data['risk'].split('(')[0].strip().upper())
    with col3:
        st.metric("Max Fridge Days", f"{int(food_data['max_fridge_days'])} days" if not pd.isna(food_data['max_fridge_days']) else "N/A")
    
    st.write(f"**Recommended Max Freezer Storage:** {int(food_data['max_freezer_months'])} months" if not pd.isna(food_data['max_freezer_months']) else "N/A")

st.header("Storage Duration")
col1, col2 = st.columns(2)

with col1:
    fridge_days = st.number_input("Days stored in fridge", min_value=0, max_value=90, value=3, step=1)
with col2:
    freezer_months = st.number_input("Months stored in freezer", min_value=0, max_value=24, value=0, step=1)

if st.button("Predict Spoilage", type="primary"):
    X = np.array([[fridge_days, freezer_months]])
    pred = loaded_nb_model.predict(X)
    pred_proba = loaded_nb_model.predict_proba(X)[0]
    
    st.divider()
    st.header("Prediction Result")
    
    if pred[0] == 1:
        st.error("*HIGH RISK - Likely SPOILED**")
        st.write(f"Confidence: {pred_proba[1]:.1%}")
    else:
        st.success("**LOW RISK - Likely SAFE**")
        st.write(f"Confidence: {pred_proba[0]:.1%}")
    
    # Show comparison with recommended values
    st.subheader("Storage Comparison")
    
    if not pd.isna(food_data['max_fridge_days']):
        max_fridge = int(food_data['max_fridge_days'])
        if fridge_days > max_fridge:
            st.warning(f"Fridge: {fridge_days} days (exceeds recommended {max_fridge} days)")
        else:
            st.info(f"✓ Fridge: {fridge_days} days (within {max_fridge} days limit)")
    
    if not pd.isna(food_data['max_freezer_months']):
        max_freezer = int(food_data['max_freezer_months'])
        if freezer_months > max_freezer:
            st.warning(f"Freezer: {freezer_months} months (exceeds recommended {max_freezer} months)")
        else:
            st.info(f"Freezer: {freezer_months} months (within {max_freezer} months limit)")
    
    st.caption("This is a prediction based on ML model. Always inspect food visually and by smell before consuming.")
