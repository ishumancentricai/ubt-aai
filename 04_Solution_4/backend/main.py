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

  return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
  """Predict an Iris class from four flower measurements."""

  feature_values = [
    features.sepal_length_cm,
    features.sepal_width_cm,
    features.petal_length_cm,
    features.petal_width_cm,
  ]
  predicted_class, probabilities = predictor.predict(feature_values)
  return PredictionResponse(
    predicted_class=predicted_class,
    probabilities=probabilities,
  )
