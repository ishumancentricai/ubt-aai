# Exercise 4: Deploying a Machine-Learning Model

## Goal

In this exercise, you will build a small but complete machine-learning deployment workflow using the Iris dataset. You will train and save a scikit-learn model, expose it through a FastAPI backend, and create a Streamlit frontend that sends measurements to the API and displays the prediction.

By the end of the exercise, you should understand how these components work together:

```text
Iris data -> trained pipeline -> saved artifact -> FastAPI -> Streamlit
```

Work through the three notebooks in order. Later parts depend on files created in earlier parts.

## Before you start: prepare the AAI environment

Exercise 4 uses additional Python packages. Install them in the existing Conda environment named `AAI` before running any notebook.

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

## Part 4.1: Train and save the model

Open `4_1_Model_Training.ipynb` and complete all six tasks marked with `YOUR CODE GOES HERE`:

1. Load and inspect the Iris dataset.
2. Create a stratified training/test split.
3. Combine `StandardScaler` and `LogisticRegression` in one scikit-learn pipeline and train it.
4. Evaluate the fitted pipeline using accuracy, a classification report, and a confusion matrix.
5. Inspect the predicted class and class probabilities for one sample.
6. Save the complete fitted pipeline and its metadata with Joblib.

The notebook must create this file before you continue:

```text
artifacts/iris_pipeline.joblib
```

This artifact is the connection between model development and the backend. It contains not only the fitted model but also the learned preprocessing and the feature/class names required during inference.

## Part 4.2: Build the FastAPI backend

Open `backend/4_2_FastAPI_Backend.ipynb`. Follow the notebook and replace every `YOUR CODE GOES HERE` section in the Python files inside `backend/`:

- `schemas.py`: define the request and response data models.
- `predictor.py`: load the saved artifact and implement prediction logic.
- `main.py`: implement the `/health` and `/predict` endpoints.
- `run.py`: configure the local Uvicorn development server.

Use the notebook tests to verify valid requests, predictions, and automatic validation of invalid data.

Then open a terminal in the `backend` folder, activate `AAI` if necessary, and start the API:

```bash
python run.py
```

Open <http://127.0.0.1:8000/docs> to test the endpoints in FastAPI's interactive documentation. Keep this terminal running while working on Part 4.3. Stop the server later with `Ctrl+C`.

## Part 4.3: Build the Streamlit frontend

Open `frontend/4_3_Streamlit_Frontend.ipynb`. Complete the marked sections in:

- `api_client.py`: send measurements to the FastAPI `/predict` endpoint and handle HTTP errors.
- `app.py`: create the four numeric inputs, submit the values, and display the predicted species and probabilities.

The FastAPI backend from Part 4.2 must still be running. Open a second terminal in the `frontend` folder, activate `AAI`, and start Streamlit:

```bash
streamlit run app.py
```

If the `streamlit` command is not found, use:

```bash
python -m streamlit run app.py
```

Streamlit normally opens <http://localhost:8501>. Enter four flower measurements and verify that the page receives a prediction from the backend.

## Completion checklist

- The `AAI` environment is active and the additional packages are installed.
- All cells in `4_1_Model_Training.ipynb` run without errors.
- `artifacts/iris_pipeline.joblib` exists.
- The backend notebook tests pass.
- `/health` and `/predict` work at <http://127.0.0.1:8000/docs>.
- The frontend sends data to the backend and displays a prediction.

## Troubleshooting

- **A package cannot be imported:** Check that `(AAI)` is visible in the terminal and that the notebook uses the `AAI` kernel. Then run the installation command again in the activated environment.
- **The model artifact cannot be found:** Finish Part 4.1 first and verify that `artifacts/iris_pipeline.joblib` was created.
- **The frontend cannot connect:** Start `python run.py` in the `backend` folder and leave that terminal open.
- **Port 8000 or 8501 is already in use:** Stop an older FastAPI or Streamlit process with `Ctrl+C` before restarting it.
- **Streamlit is not found:** Run `python -m streamlit run app.py` from the `frontend` folder.
