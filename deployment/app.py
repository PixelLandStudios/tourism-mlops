import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Setup page configuration
st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="wide")
st.title("Visit with Us: Wellness Tourism Package Prediction")
st.write("Enter the customer details below to predict if they will purchase the new Wellness Tourism Package.")

# Load the model directly from Hugging Face Model Hub
@st.cache_resource
def load_model():
    # Downloads the model from the specified repository
    model_path = hf_hub_download(repo_id="malawn/tourism-package-model", filename="best_model.joblib")
    return joblib.load(model_path)

try:
    pipeline = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Layout for user input
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=1.0, value=15.0)
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    number_of_followups = st.number_input("Number of Followups", min_value=0, value=3)

with col2:
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    number_of_trips = st.number_input("Number of Trips", min_value=1.0, value=2.0)
    passport = st.selectbox("Passport", [0, 1])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Own Car", [0, 1])

with col3:
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
    monthly_income = st.number_input("Monthly Income", min_value=1000.0, value=20000.0)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Prediction Button
if st.button("Predict Purchase"):
    # Store inputs in a DataFrame
    input_data = pd.DataFrame({
        "Age": [age],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_of_pitch],
        "NumberOfPersonVisiting": [number_of_person_visiting],
        "NumberOfFollowups": [number_of_followups],
        "PreferredPropertyStar": [preferred_property_star],
        "NumberOfTrips": [number_of_trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_satisfaction_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [number_of_children_visiting],
        "MonthlyIncome": [monthly_income],
        "TypeofContact": [typeofcontact],
        "Occupation": [occupation],
        "Gender": [gender],
        "ProductPitched": [product_pitched],
        "MaritalStatus": [marital_status],
        "Designation": [designation],
        "__index_level_0__": [0]  # Required as it was part of the training data schema
    })

    # Predict using the loaded pipeline
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"The customer is LIKELY to purchase the Wellness Tourism Package (Probability: {probability:.2%})")
    else:
        st.warning(f"The customer is UNLIKELY to purchase the Wellness Tourism Package (Probability: {probability:.2%})")
