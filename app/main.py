import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np

def get_clean_data():
    data = pd.read_csv('heart.csv')
    return data

def add_sidebar():
    st.sidebar.header("Measurements")
    data = get_clean_data()
    slider_labels = [
        ("age", "age"),
        ("sex", "sex"),
        ("cp", "cp"),
        ("trtbps", "trtbps"),
        ("chol", "chol"),
        ("fbs", "fbs"),
        ("restecg", "restecg"),
        ("thalachh", "thalachh"),
        ("exng", "exng"),
        ("oldpeak", "oldpeak"),
        ("slp", "slp"),
        ("caa", "caa"),
        ("thall", "thall"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        if key == "sex":
            input_dict[key] = st.sidebar.selectbox(
                label = 'sex',
                options=["women", "men"],
                index=0
            )
        elif key == "cp":
            input_dict[key] = st.sidebar.selectbox(
                label = 'cp',
                options=[0, 1, 2, 3],
                index=0
            )
        elif key == "fbs":
            input_dict[key] = st.sidebar.selectbox(
                label = 'fbs',
                options=[0, 1],
                index=0
            )
        elif key == "restecg":
            input_dict[key] = st.sidebar.selectbox(
                label='restecg',
                options=[0, 1],
                index=0
            )
        elif key == "exng":
            input_dict[key] = st.sidebar.selectbox(
                label='exng',
                options=[0, 1],
                index=0
            )
        elif key == "slp":
            input_dict[key] = st.sidebar.selectbox(
                label='slp',
                options=[0, 1, 2],
                index=0
            )
        elif key == "caa":
            input_dict[key] = st.sidebar.selectbox(
                label='caa',
                options=[0, 1, 2, 3],
                index=0
            )
        elif key == "thall":
            input_dict[key] = st.sidebar.selectbox(
                label='thall',
                options=[1, 2, 3],
                index=0
            )
        else:
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(0),
                max_value=float(data[key].max()),
                value=float(data[key].mean())
            )
    # Convert sex to numerical value
    input_dict["sex"] = 0 if input_dict["sex"] == "women" else 1
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['output'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    # Convert the input_dict to a DataFrame with the correct column names
    input_df = pd.DataFrame([input_data])
    input_array_scaled = scaler.transform(input_df)

    prediction = model.predict(input_array_scaled)

    st.subheader("Heart attack prediction")
    st.write("The chance of heart attack is")

    if prediction[0] == 0:
        st.write("<span class='diagnosis low'>Low</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis high'>High</span>", unsafe_allow_html=True)

    st.write("Probability of low chance of heart attack: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of high chance of heart attack: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title='Heart Attack Predictor',
        page_icon=':female-doctor:',
        layout="wide",
        initial_sidebar_state="expanded"
    )
    with open("app/assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    input_data = add_sidebar()
    with st.container():
        st.title("Lung Cancer Predictor")
        

    col1 = st.columns(1)
    with col1[0]:
        add_predictions(input_data)

if __name__ == '__main__':
    main()
