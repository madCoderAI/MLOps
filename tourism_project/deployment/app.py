import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="yuRIrocks/Tourism-Package-Prediction", filename="best_xgboost_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the type of Customer and their interaction data below to get a prediction.
""")

# User input
ProductType = st.selectbox("ProductType", ["Basic", "Deluxe", "King","Super Deluxe","Standard"])
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business","Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Designation = st.selectbox("Designation", ["Manager", "Executive","Senior Manager","AVP","VP"])
MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Divorced","Married","Unmarried"])
Age = st.number_input("Age", min_value=18, max_value=61, value=36)
NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=3, value=1)
NumberOfPersonVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=1)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=22, value=1)
MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=98678, value=1000)
OwnCar = st.selectbox("OwnCar (0: No, 1: Yes)", ["0", "1"])
Passport = st.selectbox("Passport (0: No, 1: Yes)", ["0", "1"])
CityTier = st.selectbox("CityTier", ["1", "2","3"])
PreferredPropertyStar = st.number_input("Preferred Property Rating (0 to 5)", min_value=0, max_value=5, value=4)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (0 to 10)", min_value=0, max_value=10)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=10)


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'ProductType': ProductType,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Designation': Designation,
    'MaritalStatus': MaritalStatus,
    'Age': Age,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'OwnCar': OwnCar,
    'Passport': Passport,
    'CityTier': CityTier,
    'PreferredPropertyStar': PreferredPropertyStar,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfTrips': NumberOfTrips,
    'MonthlyIncome': MonthlyIncome,
    'Gender': Gender,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfFollowups': NumberOfFollowups
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
