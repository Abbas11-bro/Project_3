import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained classifier and scaler using joblib
classifier = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the prediction function
def predict_survival(d):
    sample_data = pd.DataFrame([d])
    scaled_data = scaler.transform(sample_data)
    pred = classifier.predict(scaled_data)[0]
    prob = classifier.predict_proba(scaled_data)[0][pred]
    return pred, prob

# Streamlit UI components
st.title(" Prediction")

# Input fields for each parameter
Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=1, step=1)
Glucose = st.number_input("Glucose", min_value=0, max_value=199, value=1)
BloodPressure = st.number_input("BloodPressure", min_value=0, max_value=122, value=0)
SkinThickness = st.number_input("SkinThickness", min_value=0.0, max_value=99.0, value=1.0, step=1.0)
Insulin = st.number_input("Insulin", min_value=0, max_value=846, value=1, step=1)
BMI = st.number_input("BMI", min_value=0.0, max_value=67.1, value=7.25, step=0.5)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=2.42, value=7.25, step=0.1)


# Create the input dictionary for prediction
input_data = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
}

# When the user clicks the "Predict" button
if st.button("Predict"):
    with st.spinner('Making prediction...'):
        pred, prob = predict_survival(input_data)

        if pred == 1:
            # Survived
            st.success(f"Prediction: Diabetes with probability {prob:.2f}")
        else:
            # Not survived
            st.error(f"Prediction: Not Diabetes with probability {prob:.2f}")
