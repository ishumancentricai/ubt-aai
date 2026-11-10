"""Load the serialized pipeline and expose prediction logic."""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = (
  Path(__file__).resolve().parents[1]
  / "artifacts"
  / "iris_pipeline.joblib"
)


class IrisPredictor:
  """Small service object around the serialized scikit-learn pipeline."""

  def __init__(self, model_path: Path = MODEL_PATH):
    if not model_path.exists():
      raise FileNotFoundError(
        f"Model artifact not found at {model_path}. "
        "Run 4_1_Model_Training.ipynb first."
      )

    artifact = joblib.load(model_path)
    self.pipeline = artifact["pipeline"]
    self.feature_names = artifact["feature_names"]
    self.target_names = artifact["target_names"]

  def predict(self, feature_values: list[float]) -> tuple[str, dict[str, float]]:
    """Return the predicted class and probabilities for one sample."""

    sample = pd.DataFrame([feature_values], columns=self.feature_names)
    predicted_target = int(self.pipeline.predict(sample)[0])
    probabilities = self.pipeline.predict_proba(sample)[0]

    probability_by_class = {
      self.target_names[int(target)]: float(probability)
      for target, probability in zip(self.pipeline.classes_, probabilities)
    }
    return self.target_names[predicted_target], probability_by_class
