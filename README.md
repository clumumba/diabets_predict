Diabetes Predictor FastAPI with Docker and Data Version Control:

Files included:
- main.py — FastAPI service for diabetes risk prediction
- requirements.txt — Python runtime dependencies
- Dockerfile — container image for the service
- .dockerignore — excludes local build artifacts
- dvc/ — the diabetes prediction model project and dataset

Run locally:
1. Create and activate a virtual environment (optional but recommended):
   python -m venv .venv
   .\.venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt
3. Start the app:
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
4. Open http://127.0.0.1:8000/docs

Build and run with Docker:
1. Build the image:
   docker build -t diabetes-predictor:latest .
2. Run the container:
   docker run -p 8000:8000 diabetes-api
3. Open http://localhost:8000/docs

Example request:
- POST /predict
  {
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "SkinThickness": 29,
    "Insulin": 0,
    "BMI": 26.6,
    "DiabetesPedigreeFunction": 0.351,
    "Age": 31
  }

The service automatically trains the Random Forest model on the bundled diabetes dataset if no model file is present.
