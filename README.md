# Project Report: Interactive Regression Model

## Demo

You can view a live demo of the application here: [https://aiot-hw1-lr.streamlit.app/](https://aiot-hw1-lr.streamlit.app/)

## 1. Project Overview

This project is an interactive web application built with Streamlit that demonstrates a simple linear regression model. It allows users to dynamically generate a synthetic dataset, visualize it, train a linear regression model, and evaluate its performance. Users can also use the trained model to make predictions on new data points.

## 2. File Descriptions

-   `app.py`: The main Python script containing the Streamlit application. It handles data generation, model training, and the user interface.
-   `requirements.txt`: A text file listing the Python libraries required to run the application.
-   `GEMINI.md`: A markdown file with a brief overview of the project directory.
-   `log.md`: A markdown file, likely for logging changes or notes.
-   `.git/`: A directory indicating that this is a Git repository for version control.

## 3. `app.py`: Application Details

### 3.1. Purpose

The application serves as an educational tool to understand the fundamentals of linear regression. It provides a hands-on experience with concepts like data generation, model fitting, and evaluation.

### 3.2. Functionality

The user interface is divided into several sections:

-   **Simulation Parameters (Sidebar):** Users can adjust the following parameters to generate a dataset:
    -   **Number of Samples (n):** Controls the size of the dataset.
    -   **Coefficient (a):** Defines the slope of the true underlying linear relationship.
    -   **Noise Variance (var):** Controls the amount of random noise added to the data.
-   **1. Data Generation:** Displays the head of the generated dataset and a scatter plot of the features and target values.
-   **2. Data Preparation:** The data is split into training and testing sets, and the features are scaled using `StandardScaler`.
-   **3. Modeling:** A linear regression model is trained on the training data. The learned coefficient (slope) and intercept are displayed.
-   **4. Evaluation:** The model's performance is evaluated on the test set. The Mean Squared Error (MSE) and R-squared (R²) are displayed. A plot shows the actual vs. predicted values, with the top 5 outliers highlighted.
-   **5. Prediction:** A slider allows the user to select a new feature value and see the model's prediction for that value.

### 3.3. Libraries Used

The application relies on the following Python libraries:

-   `streamlit`: For creating the interactive web application.
-   `pandas`: For data manipulation and creating DataFrames.
-   `numpy`: For numerical operations and data generation.
-   `matplotlib`: For creating plots and visualizations.
-   `scikit-learn`: For machine learning tasks, including:
    -   `train_test_split`: For splitting data.
    -   `StandardScaler`: For feature scaling.
    -   `LinearRegression`: For the regression model.
    -   `mean_squared_error`, `r2_score`: For model evaluation.

### 3.4. Code Structure

The code is a single script that follows a logical flow:

1.  **Imports:** All necessary libraries are imported at the beginning.
2.  **Streamlit Page Configuration:** The page layout is set to "wide".
3.  **Title:** The application title is set.
4.  **Sidebar Inputs:** User-configurable parameters are defined in the sidebar.
5.  **Data Generation:** A synthetic dataset is created based on the user's inputs.
6.  **Data Visualization:** The generated data is plotted.
7.  **Data Preparation:** The data is prepared for modeling.
8.  **Model Training:** The linear regression model is trained.
9.  **Model Evaluation:** The model's performance is evaluated and visualized.
10. **Prediction:** A section for making new predictions is provided.

## 4. Dependencies

The project's dependencies are listed in `requirements.txt`:

```
pandas
numpy
matplotlib
scikit-learn
streamlit
```

## 5. How to Run the Application

1.  **Install Dependencies:** Make sure you have Python and `pip` installed. Then, install the required libraries by running the following command in your terminal:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the App:** Once the dependencies are installed, you can run the application using the following command:

    ```bash
    streamlit run app.py
    ```

3.  **View the App:** The application will open in your web browser. You can access it at `http://localhost:8501`.
