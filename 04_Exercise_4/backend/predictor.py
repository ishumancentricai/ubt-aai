"""Load the serialized pipeline and expose prediction logic."""

from pathlib import Path

import joblib


MODEL_PATH = (
  Path(__file__).resolve().parents[1]
  / "artifacts"
  / "iris_pipeline.joblib"
)


class IrisPredictor:
  """Small service object around the serialized scikit-learn pipeline."""

  def __init__(self, model_path: Path = MODEL_PATH):
    # ---- YOUR CODE GOES HERE (START) ----
    # Check that the artifact exists, load it, and store the pipeline
    # together with its feature and target names.
    raise NotImplementedError
    # ---- YOUR CODE GOES HERE (END) ----

  def predict(self, feature_values: list[float]) -> tuple[str, dict[str, float]]:
    """Return the predicted class and probabilities for one sample."""

    # ---- YOUR CODE GOES HERE (START) ----
    # Build a one-row DataFrame whose columns are self.feature_names,
    # then call predict and predict_proba and map the result to class names.
    raise NotImplementedError
    # ---- YOUR CODE GOES HERE (END) ----
