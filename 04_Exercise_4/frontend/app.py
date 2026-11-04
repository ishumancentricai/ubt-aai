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

# ---- YOUR CODE GOES HERE (START) ----
# 1. Create a form with four positive number inputs.
# 2. Build the JSON payload after the form is submitted.
# 3. Call request_prediction and handle requests.RequestException.
# 4. Display the predicted class and all class probabilities.
raise NotImplementedError
# ---- YOUR CODE GOES HERE (END) ----
