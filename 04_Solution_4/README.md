# Solution 4: Deploying a Machine-Learning Model

## Goal

This folder contains the completed reference implementation for Exercise 4. It demonstrates a small but complete machine-learning deployment workflow using the Iris dataset:

```text
Iris data -> trained pipeline -> saved artifact -> FastAPI -> Streamlit
```

The material is split into three parts that should be reviewed and run in order. The notebooks explain the implementation, while the accompanying Python modules contain the executable backend and frontend.

## Before you start: prepare the AAI environment

Exercise 4 uses additional Python packages. Install them in the existing Conda environment named `AAI` before running any notebook or application.

### macOS

1. Open a terminal. You can use the integrated terminal in VS Code via **Terminal > New Terminal**.
2. Activate the course environment:

   ```bash
   conda activate AAI
   ```

3. Check that the terminal prompt starts with `(AAI)`. This confirms that packages will be installed into the correct environment.
4. Copy the installation command from the beginning of `4_1_Model_Training.ipynb`, paste it into the terminal, and run it:

   ```bash
   pip install joblib fastapi "uvicorn[standard]" streamlit requests httpx
   ```

5. Start or select the `AAI` Jupyter kernel in VS Code before executing notebook cells.

### Windows

First try the same procedure in the integrated VS Code terminal:

```bash
conda activate AAI
pip install joblib fastapi "uvicorn[standard]" streamlit requests httpx
```

If VS Code reports that `conda` is not recognized or the environment does not activate:

1. Press the **Windows key**.
2. Search for and open **Anaconda Prompt**.
3. Activate the environment:

   ```bash
   conda activate AAI
   ```

4. Confirm that `(AAI)` appears at the beginning of the prompt.
5. Copy the installation command from `4_1_Model_Training.ipynb`, paste it into the Anaconda Prompt, and run it:

   ```bash
   pip install joblib fastapi "uvicorn[standard]" streamlit requests httpx
   ```

6. Return to VS Code and select the `AAI` environment as the notebook kernel or Python interpreter.

You normally need to install these packages only once. However, you must activate `AAI` again whenever you open a new terminal.

## Part 4.1: Model training and serialization

`4_1_Model_Training.ipynb` reviews the complete model-development workflow:

1. load and inspect the Iris dataset,
2. create a stratified training/test split,
3. combine `StandardScaler` and `LogisticRegression` in one pipeline,
4. train and evaluate the pipeline,
5. inspect a prediction and its class probabilities, and
6. serialize the fitted pipeline and its metadata with Joblib.

Running the notebook creates or replaces:

```text
artifacts/iris_pipeline.joblib
```

The artifact includes the learned preprocessing, classifier, feature names, and target names. Keeping these elements together ensures that the backend uses exactly the same transformation as the training code.

## Part 4.2: FastAPI backend

`backend/4_2_FastAPI_Backend.ipynb` explains and tests the completed backend modules:

- `schemas.py` defines and validates the HTTP request and response formats.
- `predictor.py` loads the Joblib artifact and converts incoming measurements into model predictions.
- `main.py` defines the `/health` and `/predict` routes.
- `run.py` starts the application with Uvicorn on port 8000.

To run the backend, open a terminal in the `backend` folder, activate `AAI` if necessary, and execute:

```bash
python run.py
```

The API is available at <http://127.0.0.1:8000>, and FastAPI's interactive documentation is available at <http://127.0.0.1:8000/docs>. Keep the terminal running for the frontend demonstration. Stop the server with `Ctrl+C` when finished.

## Part 4.3: Streamlit frontend

`frontend/4_3_Streamlit_Frontend.ipynb` explains the completed frontend:

- `api_client.py` sends an HTTP POST request to the FastAPI `/predict` endpoint and handles request errors.
- `app.py` collects four Iris measurements, submits them to the API, and displays the predicted species and all class probabilities.

Leave the backend running. Open a second terminal in the `frontend` folder, activate `AAI`, and run:

```bash
streamlit run app.py
```

If the `streamlit` command is not found, use:

```bash
python -m streamlit run app.py
```

Streamlit normally opens <http://localhost:8501>. A successful prediction confirms that the browser-facing frontend, HTTP client, API, and serialized model work together.

## Recommended review workflow

1. Run `4_1_Model_Training.ipynb` from top to bottom and inspect the generated artifact.
2. Read the modules in `backend/` and run the tests in `4_2_FastAPI_Backend.ipynb`.
3. Start the backend and test both endpoints at <http://127.0.0.1:8000/docs>.
4. Read the modules in `frontend/` and run the API-client test in `4_3_Streamlit_Frontend.ipynb`.
5. Start Streamlit in a second terminal and submit example measurements.
6. Compare this implementation with your exercise files and identify differences in validation, error handling, and separation of responsibilities.

## Troubleshooting

- **A package cannot be imported:** Check that `(AAI)` is visible in the terminal and that the notebook uses the `AAI` kernel. Then run the installation command again in the activated environment.
- **The model artifact cannot be found:** Run `4_1_Model_Training.ipynb` completely before starting the backend.
- **The frontend cannot connect:** Start `python run.py` in the `backend` folder and leave that terminal open.
- **Port 8000 or 8501 is already in use:** Stop an older FastAPI or Streamlit process with `Ctrl+C` before restarting it.
- **Streamlit is not found:** Run `python -m streamlit run app.py` from the `frontend` folder.
