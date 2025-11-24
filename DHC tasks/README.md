# House Price Prediction Streamlit App

## Overview

This is a Streamlit web application for predicting house prices based on property features. It uses machine learning models (Linear Regression and Gradient Boosting) trained on a dataset of house features.

The app allows you to:

* View the **full dataset**
* Train and evaluate regression models
* See **MAE and RMSE** for model evaluation
* **Manually input house features** and get predicted prices

---

## Features

* Preprocessing of numeric and categorical features
* Model training with Linear Regression and Gradient Boosting Regressor
* Full dataset display in the web app
* Manual input form for predicting house prices
* Evaluation metrics (MAE and RMSE)

---

## Dataset

* The app requires a CSV file with house data, including features such as:

  * `Area`
  * `Bedrooms`
  * `Bathrooms`
  * `Floors`
  * `YearBuilt`
  * `Location`
  * `Condition`
  * `Garage`
  * `Price` (target variable)

* You can use the **House Price Prediction Dataset from Kaggle**.

* Place the CSV in the project folder as `house_price_data.csv` or upload it via the app interface.

---

## How to Run

1. Install required packages:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

2.Place your CSV file in the project folder as `house_price_data.csv` or upload via the app.

3.Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

4.The app will open in your browser:

* View the **full dataset**
* Check **model evaluation metrics**
* Enter house features in the manual input form to get predicted prices

---

## Usage

* Use the numeric inputs for features like Area, Bedrooms, Bathrooms, Floors, and YearBuilt.
* Use the dropdowns for categorical features like Location, Condition, and Garage.
* Click **Predict Price** to see predictions from both models.

---

## Evaluation Metrics

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**

Both metrics are displayed in the app for each model.

---

## Notes

* Ensure that your CSV dataset has consistent column names.
* The app automatically scales numeric features and encodes categorical variables for prediction.
* The manual input form aligns with the training data to prevent feature mismatch errors.

---

## Author

Developed by Nawal Shahid for DevelopersHub Corporation Internship.
