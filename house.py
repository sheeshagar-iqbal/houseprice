import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("model.pkl")  
try:
    scaler = joblib.load("scaler.pkl")  
except:
    scaler = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.write("Enter the details of the house to estimate its price.")

col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Square Footage", min_value=200, max_value=20000, value=1500, step=10)
    num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=20, value=3, step=1)
    num_bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=2005, step=1)

with col2:
    lot_size = st.number_input("Lot Size (acres)", min_value=0.0, max_value=100.0, value=0.25, step=0.01, format="%.2f")
    garage_size = st.number_input("Garage Size (car spaces)", min_value=0, max_value=5, value=1, step=1)
    neighborhood_quality = st.number_input("Neighborhood Quality", min_value=1,max_value=10,value=5,step=1)



# Create dataframe for prediction
features = pd.DataFrame([[
    square_footage, num_bedrooms, num_bathrooms,
    year_built, lot_size, garage_size, neighborhood_quality
]], columns=[
    "Square_Footage", "Num_Bedrooms", "Num_Bathrooms",
    "Year_Built", "Lot_Size", "Garage_Size", "Neighborhood_Quality"
])

st.subheader("Your Input Data")
st.dataframe(features,use_container_width=True)

# Prediction button
if st.button("🔮 Predict House Price"):
    X = features.copy()

    # Apply scaler if available
    if scaler is not None:
        X = scaler.transform(X)

    prediction = model.predict(X)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")
