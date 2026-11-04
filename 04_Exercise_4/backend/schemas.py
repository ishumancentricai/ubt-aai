"""Request and response schemas for the prediction API."""

from pydantic import BaseModel


class IrisFeatures(BaseModel):
  """Define the four positive flower measurements expected by the model."""

  # ---- YOUR CODE GOES HERE (START) ----
  pass
  # ---- YOUR CODE GOES HERE (END) ----


class PredictionResponse(BaseModel):
  """Define the predicted class and per-class probabilities."""

  # ---- YOUR CODE GOES HERE (START) ----
  pass
  # ---- YOUR CODE GOES HERE (END) ----
