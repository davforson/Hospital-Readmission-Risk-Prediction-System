from fastapi import FastAPI, HTTPException
from api.schemas import PatientFeatures, PredictionResponse, HealthResponse
import torch
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Readmission Prediction API",
    description="Predict 30-day hospital readmission risk",
    version="1.0.0",
)

# -- Global Mode State -- Loaded once at startup
model = None
scaler = None
feature_cols = None
model_version = "unknown"


@app.on_event("startup")
async def load_model():
    "Load model, scaler and feature list when the server starts"
    global model, scaler, feature_cols, model_version

    from src.model.architecture import ReadmissionPredictor

    # Load the features from the json file
    import json

    with open("data/processed/feature_cols.json", "r") as f:
        feature_cols = json.load(f)

    # Load scaler
    scaler = torch.load("data/processed/scaler.pt", weights_only=False)

    # Load model:
    model = ReadmissionPredictor(
        input_dim=len(feature_cols), hidden_dims=[64, 32], dropout_rate=0.2
    )
    model.load_state_dict(torch.load("data/processed/model.pt", weights_only=True))
    model.eval()

    model_version = "v2-staging"

    logger.info(f"Model loaded: {len(feature_cols)} features, version {model_version}")


def prepare_features(patient: PatientFeatures) -> np.ndarray:
    """Convert API input into the feature array the model expects.

    This handles one-hot encoding for categorical fields
    so the API consumer sends human-readable strings.
    """
    # Start with a dict of raw values

    data = {
        "length_of_stay": patient.length_of_stay,
        "number_of_procedures": patient.number_of_procedures,
        "number_of_diagnoses": patient.number_of_diagnoses,
        "num_prior_admissions_6mo": patient.num_prior_admissions_6mo,
        "num_prior_admissions_12mo": patient.num_prior_admissions_12mo,
        "days_since_last_admission": patient.days_since_last_admission,
        "total_previous_los": patient.total_previous_los,
        "avg_glucose": patient.avg_glucose,
        "avg_creatinine": patient.avg_creatinine,
        "avg_hemoglobin": patient.avg_hemoglobin,
        "avg_white_blood_cell": patient.avg_white_blood_cell,
        "avg_bun": patient.avg_bun,
        "avg_sodium": patient.avg_sodium,
        "avg_potassium": patient.avg_potassium,
        "avg_hba1c": patient.avg_hba1c,
        "num_lab_tests": patient.num_lab_tests,
        "age_at_admission": patient.age_at_admission,
    }

    # One-hot encode categorical fields
    # Gender (drop_first=True means we drop "Female" as baseline)
    data["gender_Male"] = 1 if patient.gender == "Male" else 0
    data["gender_Other"] = 1 if patient.gender == "Other" else 0

    # Insurance (drop_first=True means we drop "Medicaid" as baseline)
    for ins_type in ["Medicare", "Private", "Self-Pay", "VA"]:
        data[f"insurance_{ins_type}"] = 1 if patient.insurance_type == ins_type else 0

    # Admission type (drop_first=True means we drop "Elective" as baseline)
    for adm_type in ["Emergency", "Trauma", "Urgent"]:
        data[f"admission_{adm_type}"] = 1 if patient.admission_type == adm_type else 0

    # Discharge disposition (drop_first=True means we drop "AMA" as baseline)
    for disp in ["Expired", "Home", "Home Health", "Rehab", "SNF"]:
        data[f"discharge_{disp}"] = 1 if patient.discharge_disposition == disp else 0

    for col in feature_cols:
        if col.startswith("diag") and col not in data:
            code = col.replace("diag", "")
            data[col] = 1 if patient.primary_diagnosis_code == code else 0

    # Build the array in the exact column order the feature expects
    feature_array = np.array([[data.get(col, 0) for col in data]], dtype=np.float32)

    # Scale using the training scaler
    feature_array = scaler.transform(feature_array)

    return feature_array


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient: PatientFeatures):
    """Get readmission risk prediction for a patient."""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        feature_array = prepare_features(patient)

        with torch.no_grad():
            tensor = torch.tensor(feature_array)
            logit = model(tensor)
            probability = torch.sigmoid(logit).item()

        # Classify risk level
        if probability >= 0.7:
            risk_level = "high"
        elif probability >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return PredictionResponse(
            readmission_probability=round(probability, 4),
            risk_level=risk_level,
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running and model is loaded."""
    return HealthResponse(
        status="healthy", model_loaded=model is not None, model_version=model_version
    )
