import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score
import matplotlib.pyplot as plt
import streamlit as st


data = {
    "area": [1000,1500,1800,2400,3000,1200,2000,2200,2700,3200],
    "bedrooms": [2,3,3,4,4,2,3,3,4,5],
    "age": [20,15,10,5,2,18,8,6,3,1],
    "price": [200000,300000,350000,500000,650000,220000,400000,450000,600000,700000]
}

df =pd.DataFrame(data)

st.title("House Price Predictor 🏠")

X =df[["area","bedrooms","age"]]
y= df["price"]



area = st.sidebar.number_input("Area (sq ft)", 500, 5000, 1500)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
age = st.sidebar.number_input("Age of house (years)", 0, 50, 10)


X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.2,random_state=42)

model =LinearRegression()
model.fit(X_train,y_train)

y_pred =model.predict(X_test)

st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R2:", r2_score(y_test, y_pred))

print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)

print("Actual :",y_test.values)
print("Predicted:",y_pred)

if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame([[area, bedrooms, age]], columns=["area","bedrooms","age"])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ₹{prediction[0]:,.0f}")

new_house = [[2560,6,3]]
prediction = model.predict(new_house)

print("Predicted Price of new House -",prediction)

st.subheader("Area vs Price")
plt.scatter(df["area"], df["price"])
plt.xlabel("Area")
plt.ylabel("Price")
st.pyplot(plt)
