import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1️⃣ Load dataset
df = pd.read_csv("diabetes997.csv")
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']

# 2️⃣ Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3️⃣ Streamlit UI
st.title("Diabetes Predictor")
st.write("Enter patient data below to predict diabetes:")

Pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
Glucose = st.number_input("Glucose", min_value=0, value=120)
BloodPressure = st.number_input("Blood Pressure", min_value=0, value=70)
SkinThickness = st.number_input("Skin Thickness", min_value=0, value=20)
Insulin = st.number_input("Insulin", min_value=0, value=0)
BMI = st.number_input("BMI", min_value=0.0, value=25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
Age = st.number_input("Age", min_value=0, value=30)

# 4️⃣ Predict button
if st.button("Predict"):
    new_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
                            columns=X.columns)
    prediction = model.predict(new_data)[0]
    result = "Diabetic" if prediction == 1 else "Non-diabetic"
    st.success(f"Prediction Result: {result}")