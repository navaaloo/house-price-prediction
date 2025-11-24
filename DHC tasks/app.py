# streamlit_app.py
# ---------------------------------------------------------------
# House Price Prediction Streamlit App with Manual Feature Input
# Fixed feature alignment issue for prediction
# ---------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------
st.set_page_config(page_title="House Price Prediction App", page_icon="🏠", layout="wide")
st.title("🏠 House Price Prediction App")
st.write("Upload your dataset or use the local CSV to predict house prices using ML models.")

# ---------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------
local_file = "house_price_data.csv"
try:
    data = pd.read_csv(local_file)
    st.success(f"Loaded local CSV file: {local_file}")
except FileNotFoundError:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Please upload a CSV file to begin or place it in the project folder as 'house_price_data.csv'.")
        st.stop()

# ---------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------
st.subheader("📌 Dataset Preview")
st.dataframe(data)

# ---------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------
features = ['Area','Bedrooms','Bathrooms','Floors','YearBuilt','Location','Condition','Garage']
target = 'Price'

# Encode categorical features
data = pd.get_dummies(data, columns=['Location','Condition','Garage'], drop_first=True)
data = data.dropna()

# Separate X and y
X = data.drop(target, axis=1)
y = data[target]

# Scale numeric features
numeric_cols = ['Area','Bedrooms','Bathrooms','Floors','YearBuilt']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------
# Train models
# ---------------------------------------------------------------
lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)
lin_mae = mean_absolute_error(y_test, lin_pred)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))

gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_pred = gbr.predict(X_test)
gbr_mae = mean_absolute_error(y_test, gbr_pred)
gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_pred))

# ---------------------------------------------------------------
# Show model evaluation
# ---------------------------------------------------------------
st.subheader("📊 Model Performance Metrics")
col1, col2 = st.columns(2)
with col1:
    st.write("### Linear Regression")
    st.write(f"MAE: {lin_mae:.2f}")
    st.write(f"RMSE: {lin_rmse:.2f}")
with col2:
    st.write("### Gradient Boosting")
    st.write(f"MAE: {gbr_mae:.2f}")
    st.write(f"RMSE: {gbr_rmse:.2f}")

# ---------------------------------------------------------------
# Manual Input for Prediction
# ---------------------------------------------------------------
st.subheader("🖊 Enter House Features for Price Prediction")

# Numeric inputs
area = st.number_input("Area (sqft)", min_value=100.0, value=1000.0)
bedrooms = st.number_input("Bedrooms", min_value=1, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
floors = st.number_input("Floors", min_value=1, value=1)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)

# Categorical inputs
location = st.selectbox("Location", options=['Urban','Rural','Downtown' , 'Suburban'])
condition = st.selectbox("Condition", options=['Good','Excellent','Poor' , 'Fair'])
garage = st.selectbox("Garage", options=['Yes','No'])

if st.button("Predict Price"):
    # Create input dict
    input_dict = {'Area': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
                  'Floors': floors, 'YearBuilt': year_built}

    # One-hot encode categorical features based on training columns
    for col in X.columns:
        if 'Location_' in col:
            input_dict[col] = 1 if col == f'Location_{location}' else 0
        elif 'Condition_' in col:
            input_dict[col] = 1 if col == f'Condition_{condition}' else 0
        elif 'Garage_' in col:
            input_dict[col] = 1 if col == f'Garage_{garage}' else 0
        elif col not in input_dict:
            input_dict[col] = 0

    input_df = pd.DataFrame([input_dict])

    # Ensure numeric columns are scaled
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Reorder columns to match training data
    input_df = input_df[X.columns]

    # Predict
    lin_price = lin.predict(input_df)[0]
    gbr_price = gbr.predict(input_df)[0]

    st.write(f"**Linear Regression Predicted Price:** {lin_price:,.2f}")
    st.write(f"**Gradient Boosting Predicted Price:** {gbr_price:,.2f}")

st.success("App is ready! Enter features and predict house price.")
