"""FastAPI application exposing the Iris prediction pipeline."""

from fastapi import FastAPI

from predictor import IrisPredictor
from schemas import IrisFeatures, PredictionResponse


app = FastAPI(
  title="Iris Prediction API",
  description="Minimal FastAPI backend for a trained scikit-learn pipeline.",
  version="1.0.0",
)
predictor = IrisPredictor()


@app.get("/health")
def health() -> dict[str, str]:
  """Report whether the API and model loaded successfully."""

  # ---- YOUR CODE GOES HERE (START) ----
  raise NotImplementedError
  # ---- YOUR CODE GOES HERE (END) ----


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
  """Predict an Iris class from four flower measurements."""

  # ---- YOUR CODE GOES HERE (START) ----
  raise NotImplementedError
  # ---- YOUR CODE GOES HERE (END) ----
