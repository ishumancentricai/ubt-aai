"""Request and response schemas for the prediction API."""

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
  """Four measurements expected by the trained Iris pipeline."""

  sepal_length_cm: float = Field(gt=0, description="Sepal length in cm")
  sepal_width_cm: float = Field(gt=0, description="Sepal width in cm")
  petal_length_cm: float = Field(gt=0, description="Petal length in cm")
  petal_width_cm: float = Field(gt=0, description="Petal width in cm")


class PredictionResponse(BaseModel):
  """Prediction and probability assigned to each Iris class."""

  predicted_class: str
  probabilities: dict[str, float]
