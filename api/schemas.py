from pydantic import BaseModel, Field
from typing import List, Optional

class PatientFeatures(BaseModel):
    """Input schema: features needed for a readmission prediction.

    Every field has:
    - A type (float, int, str)
    - A Field() with constraints (ge=0 means >= 0)
    - A description (shows up in the auto-generated docs)

    Pydantic validates ALL of these automatically.
    Send a negative value for length_of_stay? → instant 422 error.
    """
    # Clinical features
    length_of_stay: int = Field(..., ge=1, description="Days in the hospital(minimum 1)")
    number_of_procedures: int= Field(..., ge=0, descripton="Procedures performed")
    number_of_diagnoses: int= Field(..., ge=1, description="Diagnosis count")
    primary_diagnosis_code: str= Field(..., description="ICD-10 code, e.g. I50.9")


    # Utilization Features
    num_prior_admissions_6mo: int = Field(..., ge=0, description="Admissions in the last 6 months")
    num_prior_admissions_12mo: int = Field(..., ge=0, description="Admissions in the last 12 months")
    days_since_last_admission: int = Field(..., ge=0, description="Days since last discharge")
    total_previous_los: int = Field(..., ge=0, description="Total prior length of stay")

    # Lab features
    avg_glucose: float = Field(0.0, ge=0, description="Average glucose level")
    avg_creatinine: float = Field(0.0, ge=0, description="Average creatinine level")
    avg_hemoglobin: float = Field(0.0, ge=0, description="Average hemoglobin level")
    avg_white_blood_cell: float = Field(0.0, ge=0, description="Average WBC count")
    avg_bun: float = Field(0.0, ge=0, description="Average BUN level")
    avg_sodium: float = Field(0.0, ge=0, description="Average sodium level")
    avg_potassium: float = Field(0.0, ge=0, description="Average potassium level")
    avg_hba1c: float = Field(0.0, ge=0, description="Average HbA1c level")
    num_lab_tests: int = Field(0, ge=0, description="Total lab tests performed")

    # Demographic features
    age_at_admission: int = Field(..., ge=0, le=120, description="Patient age")
    gender: str = Field(..., description="Male, Female, or Other")
    insurance_type: str = Field(..., description="Medicare, Medicaid, Private, Self-Pay, or VA")

    # Admission/discharge details
    admission_type: str = Field(..., description="Emergency, Urgent, Elective, or Trauma")
    discharge_disposition: str = Field(..., description="Home, SNF, Rehab, Home Health, AMA, or Expired")


class PredictionResponse(BaseModel):
    """Output Schema: what the API returns

    """
    readmission_probability: float = Field(..., description="Probability of 30-day readmission (0 to 1)")
    risk_level: str = Field(..., description="low, medium, or high")
    model_version: str = Field(..., description="Which model version made this prediction")


class HealthResponse(BaseModel):
    """Health Check Response"""

    status: str
    model_loaded: bool
    model_version: str

