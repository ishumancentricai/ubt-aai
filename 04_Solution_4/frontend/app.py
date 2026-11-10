"""Streamlit UI for the Iris prediction API."""

import requests
import streamlit as st

from api_client import request_prediction


st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("Iris Classifier")
st.write(
  "Enter four flower measurements. The frontend sends them to the "
  "FastAPI backend and displays its prediction."
)

with st.form("iris_features"):
  sepal_length = st.number_input(
    "Sepal length (cm)", min_value=0.1, value=5.1, step=0.1
  )
  sepal_width = st.number_input(
    "Sepal width (cm)", min_value=0.1, value=3.5, step=0.1
  )
  petal_length = st.number_input(
    "Petal length (cm)", min_value=0.1, value=1.4, step=0.1
  )
  petal_width = st.number_input(
    "Petal width (cm)", min_value=0.1, value=0.2, step=0.1
  )
  submitted = st.form_submit_button("Predict")

if submitted:
  payload = {
    "sepal_length_cm": sepal_length,
    "sepal_width_cm": sepal_width,
    "petal_length_cm": petal_length,
    "petal_width_cm": petal_width,
  }

  try:
    result = request_prediction(payload)
  except requests.RequestException as error:
    st.error(
      "The FastAPI backend could not be reached. "
      "Start it from the backend folder with `python run.py`."
    )
    st.exception(error)
  else:
    st.success(f"Predicted class: {result['predicted_class']}")
    st.subheader("Class probabilities")
    for class_name, probability in result["probabilities"].items():
      st.write(f"{class_name}: {probability:.1%}")
      st.progress(probability)
