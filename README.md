# streamlit_heart_attack_prediction
Heart Attack Predictor

This project is a web application designed to predict the risk of a heart attack based on user input. The application utilizes a machine learning model to provide predictions and is built using Streamlit.

Features:

User-friendly interface for inputting personal health data

Real-time prediction of heart attack risk

Displays probabilities of low and high risk

Supports medical professionals in making informed decisions

Installation

Clone the repository:
git clone https://github.com/your-repo/heart-attack-predictor.git

Navigate to the project directory:

cd heart-attack-predictor

Install the required dependencies:

pip install -r requirements.txt

Usage:

Ensure you have the heart dataset (heart.csv) in the project directory.

Place the trained model (model.pkl) and the scaler (scaler.pkl) in the model directory.

Run the Streamlit application:
streamlit run main.py
Open your web browser and go to http://localhost:8501 to access the application.

Data Input:

The application requires the following health measurements to predict heart attack risk:

Age

Sex

Chest pain type (cp)

Resting blood pressure (trtbps)

Cholesterol level (chol)

Fasting blood sugar (fbs)

Resting electrocardiographic results (restecg)

Maximum heart rate achieved (thalachh)

Exercise-induced angina (exng)

ST depression induced by exercise relative to rest (oldpeak)

The slope of the peak exercise ST segment (slp)

Number of major vessels (caa)

Thalassemia (thall)

How It Works

Data Input: Users input their health measurements via the sidebar.

Data Scaling: The input data is scaled using the pre-trained scaler.

Prediction: The scaled data is fed into the pre-trained machine learning model to get the prediction.

Output: The application displays the risk of heart attack as either low or high, along with the probabilities.
Disclaimer

This application is intended to assist medical professionals in making a diagnosis. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

https://appheartattackprediction-8rj5cwqnuqnrnogpzjtmeq.streamlit.app/
