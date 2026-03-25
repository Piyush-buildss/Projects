import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- Load or create your dataset ---
data = pd.read_csv("data.csv")
X = data[["hours_study","attendance","number_courses"]]
Y = data[["math_marks","science_marks","english_marks"]]

# --- Train model ---
model = LinearRegression()
model.fit(X, Y)

# --- Streamlit UI ---
st.title("Student Marks Prediction")

hours = st.number_input("Hours of Study", min_value=0, max_value=24, value=5)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=90)
num_courses = st.number_input("Number of Courses", min_value=1, max_value=10, value=6)

if st.button("Predict Marks"):
    new_student = pd.DataFrame([[hours, attendance, num_courses]],
                               columns=["hours_study","attendance","number_courses"])
    pred = model.predict(new_student)
    st.write(f"Predicted Math Marks: {pred[0][0]:.2f}")
    st.write(f"Predicted Science Marks: {pred[0][1]:.2f}")
    st.write(f"Predicted English Marks: {pred[0][2]:.2f}")