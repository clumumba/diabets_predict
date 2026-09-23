# Diabetes Predictor

FastAPI inference service and DVC-managed training pipeline for the Diabetes Predictor project.

Files included:
- main.py — FastAPI service for diabetes risk prediction
- requirements.txt — Python runtime dependencies
- Dockerfile — container image for the service
- .dockerignore — excludes local build artifacts
- dvc/ — the diabetes prediction model project and dataset

Run locally from the `dvc` directory:
1. Create and activate a virtual environment (optional but recommended):
   python -m venv .venv
   .\.venv\Scripts\activate
2. Install dependencies:
   pip install -r requirements.txt
3. Reproduce the training pipeline (creates `models/model.pkl`):
   dvc repro
4. Start the app:
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
5. Open http://127.0.0.1:8000/docs

Build and run with Docker:
1. Build the image:
   docker build -t clumumba62/diabetes-predictor:latest .
2. Run the container:
   docker run --rm -p 8000:8000 clumumba62/diabetes-predictor:latest
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

The API loads the model artifact produced by DVC at startup. It intentionally fails startup if `models/model.pkl` is missing; run `dvc repro` before starting the API or building the image. The service returns a model score and demonstration risk bucket, and is not a diagnostic or clinically validated system.

## CI artifact requirement

The image-publishing job restores DVC artifacts from the DagsHub remote, builds the image, and checks `/health` for `model_loaded: true` before pushing tags. Configure these GitHub repository secrets before the workflow can publish:

- Secret `DAGSHUB_TOKEN`: a DagsHub token that can read this repository's DVC artifacts. Do not commit it to Git.
- Secrets `DOCKER_USERNAME` and `DOCKER_PASSWD`: Docker Hub publishing credentials.

Configure a local DagsHub token and upload the DVC-tracked raw data and model artifacts before the first CI run. The token is entered at a secure prompt and saved only in `.dvc/config.local`:

```powershell
$secureToken = Read-Host "DagsHub token" -AsSecureString
$dagshubToken = [System.Net.NetworkCredential]::new("", $secureToken).Password
dvc remote modify --local origin access_key_id $dagshubToken
dvc remote modify --local origin secret_access_key $dagshubToken
dvc push -r origin
Remove-Variable dagshubToken, secureToken
```

Add the replacement token as the GitHub `DAGSHUB_TOKEN` secret after rotating the exposed token. The workflow fails before publishing when the token is missing or when the restored model is absent. The Docker build copies only the API code and model, leaving the DVC cache, source data, and local DVC configuration out of the image.
