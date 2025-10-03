import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

st.title("Interactive Regression Model with Streamlit")

# Sidebar for user inputs
st.sidebar.header("Simulation Parameters")

n_samples = st.sidebar.slider(
    "Number of Samples (n)",
    min_value=100,
    max_value=5000,
    value=200,
    step=100
)

coefficient_a = st.sidebar.slider(
    "Coefficient (a)",
    min_value=-10.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)

noise_variance = st.sidebar.slider(
    "Noise Variance (var)",
    min_value=0,
    max_value=1000,
    value=15,
    step=10
)

# Generate synthetic dataset
st.header("1. Data Generation")
st.write(f"Generating a dataset with {n_samples} samples, a true coefficient of {coefficient_a}, and noise variance of {noise_variance}.")

X = np.random.rand(n_samples, 1) * 10 # Features between 0 and 10
true_y = coefficient_a * X.flatten()
noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)
y = true_y + noise

df = pd.DataFrame({'Feature': X.flatten(), 'Target': y})

st.dataframe(df.head())

fig_data, ax_data = plt.subplots()
ax_data.scatter(df['Feature'], df['Target'], alpha=0.6)
ax_data.set_title("Data Understanding - Scatter Plot")
ax_data.set_xlabel("Feature")
ax_data.set_ylabel("Target")
st.pyplot(fig_data)

# Data Preparation
st.header("2. Data Preparation")
X_train, X_test, y_train, y_test = train_test_split(
    df[['Feature']], df['Target'], test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
st.write("Data split into training and testing sets, and features scaled.")

# Modeling
st.header("3. Modeling")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

st.write(f"Coefficient (slope): {model.coef_[0]:.2f}")
st.write(f"Intercept: {model.intercept_:.2f}")

# Generated Data and Linear Regression
st.header("Generated Data and Linear Regression")
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R²): {r2:.2f}")

fig_eval, ax_eval = plt.subplots()
ax_eval.scatter(X_test_scaled, y_test, color="blue", label="Actual")
ax_eval.plot(X_test_scaled, y_pred, color="red", linewidth=2, label="Predicted")
ax_eval.set_title("Evaluation - Linear Regression Fit")
ax_eval.legend()

# Label top 5 outliers
num_outliers_to_label = 5
residuals = y_test - y_pred
abs_residuals = np.abs(residuals)
# Get indices of top outliers
outlier_indices = np.argsort(abs_residuals)[-num_outliers_to_label:]

for idx, i in enumerate(outlier_indices):
    ax_eval.scatter(X_test_scaled[i], y_test.iloc[i], color='purple', s=150, marker='D', zorder=5)
    ax_eval.annotate(
        f'Sample {i}', # Display sample number
        (X_test_scaled[i], y_test.iloc[i]),
        textcoords="offset points",
        xytext=(0,15), # Adjust offset for better visibility
        ha='center',
        fontsize=10, # Slightly larger font for numbers
        color='purple'
    )

st.pyplot(fig_eval)


# Deployment (Simulation)
st.header("5. Prediction")
new_feature_value = st.slider(
    "Select a feature value for prediction",
    float(df['Feature'].min()),
    float(df['Feature'].max()),
    float(df['Feature'].mean())
)
new_data = np.array([[new_feature_value]])
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
st.write(f"Predicted target for feature={new_feature_value:.2f}: {prediction[0]:.2f}")