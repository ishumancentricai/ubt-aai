"""HTTP client used by the Streamlit user interface."""

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"


def request_prediction(
  payload: dict[str, float],
  api_url: str = DEFAULT_API_URL,
  timeout: float = 5.0,
) -> dict:
  """Send one prediction request and return the decoded JSON response."""

  # ---- YOUR CODE GOES HERE (START) ----
  raise NotImplementedError
  # ---- YOUR CODE GOES HERE (END) ----
