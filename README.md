# 🏥 Hospital Readmission Risk Prediction System

An end-to-end machine learning system that predicts 30-day hospital readmissions, built with production-grade MLOps practices. The project covers the full ML lifecycle — from data extraction to model deployment and monitoring — using the same tools and patterns used by top healthcare AI teams.

---

## 📌 The Business Problem

In the United States, the Centers for Medicare & Medicaid Services (CMS) penalizes hospitals with above-average 30-day readmission rates through the Hospital Readmissions Reduction Program (HRRP). Penalties can reach up to 3% of total Medicare reimbursements — translating to millions of dollars annually for large hospitals.

This system predicts which patients are likely to be readmitted within 30 days of discharge, enabling care teams to intervene proactively with follow-up appointments, home health services, and medication reconciliation before the patient returns to the hospital.

---

## 🏗️ System Architecture

```
DATA SOURCES              PIPELINE                    SERVING & MONITORING
──────────────────────────────────────────────────────────────────────────

┌──────────────┐
│ PostgreSQL   │──┐
└──────────────┘  │
┌──────────────┐  │   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ CSV Files    │──┼──→│ Validate │──→│Transform │──→│  Train   │
└──────────────┘  │   │  (GX)    │   │ (pandas) │   │(PyTorch) │
┌──────────────┐  │   └──────────┘   └──────────┘   └────┬─────┘
│ Meds API     │──┘                                      │
└──────────────┘                                         │
                                                         │
          ┌──────────────────────────────────────────────┘
          │
          ├───→ MLflow (track experiments, version models)
          ├───→ FastAPI (serve predictions via REST API)
          ├───→ Streamlit (monitoring + clinician prediction UI)
          │
     ┌────┴─────┐
     │ Airflow  │  ← orchestrates the pipeline on schedule
     └──────────┘
          │
     ┌────┴─────┐
     │  Docker  │  ← packages all services into containers
     └──────────┘
          │
     ┌────┴──────────┐
     │ GitHub Actions │  ← automated testing & deployment
     └───────────────┘
```

---

## 🛠️ Tools & Technologies

| Tool | Purpose | Section |
|------|---------|---------|
| **PostgreSQL** | Patient records, admissions, diagnoses | Data Extraction |
| **pandas** | Data manipulation, cleaning, feature engineering | Transformation |
| **Great Expectations** | Automated data quality validation | Validation |
| **Apache Airflow** | Pipeline orchestration and scheduling | Orchestration |
| **PyTorch** | Neural network model training | Model Training |
| **MLflow** | Experiment tracking and model registry | Experiment Tracking |
| **FastAPI** | REST API for real-time predictions | Model Serving |
| **Streamlit** | Monitoring dashboard + clinician prediction UI | Monitoring & UI |
| **Docker** | Containerization of all services | Deployment |
| **GitHub Actions** | CI/CD pipeline (testing, linting, deployment) | CI/CD |

---

## 📂 Project Structure

```
readmission-prediction/
├── .github/workflows/
│   ├── ci.yml                        # Lint, test, format on every push/PR
│   └── cd.yml                        # Build and push Docker image on merge
├── airflow/dags/
│   ├── readmission_etl_dag.py        # Daily ETL: extract → validate → transform
│   └── readmission_train_dag.py      # Daily training: train → evaluate → register
├── api/
│   ├── main.py                       # FastAPI prediction endpoints
│   └── schemas.py                    # Pydantic request/response models
├── dashboard/
│   ├── app.py                        # Model monitoring (drift, quality, performance)
│   └── predict.py                    # Clinician-facing prediction interface
├── data/
│   ├── raw/                          # Extracted, untouched data (ELT staging)
│   └── processed/                    # Cleaned features, trained model, scaler
├── scripts/
│   └── setup_database.py             # Synthetic data generation and DB setup
├── src/
│   ├── extraction/                   # Database, CSV, and API extractors
│   ├── validation/                   # Great Expectations data contracts
│   ├── transformation/               # Data cleaning and feature engineering
│   ├── model/                        # PyTorch architecture, training, evaluation
│   └── monitoring/                   # Drift detection utilities
├── tests/
│   └── test_model.py                 # Model architecture tests
├── Dockerfile                        # API container image
├── docker-compose.yml                # Multi-service orchestration
├── requirements.txt                  # Python dependencies (pinned versions)
└── requirements-docker.txt           # Linux-compatible dependencies for Docker
```

---

## 🎯 ML Pipeline

### Data Sources
| Source | Type | Contents |
|--------|------|----------|
| PostgreSQL | Database | Patient demographics, admission records, ICD-10 diagnosis codes |
| lab_results.csv | Static file | Lab tests: glucose, creatinine, hemoglobin, WBC, sodium, potassium, BUN, HbA1c |
| Medications API | REST API | Prescription history, drug interactions, adherence scores |

### Feature Engineering (33 features)
- **Clinical**: length of stay, number of procedures, number of diagnoses, diagnosis codes
- **Utilization**: prior admissions (6mo/12mo), days since last admission, total previous LOS
- **Lab results**: average values for 8 lab tests, total lab test count
- **Demographics**: age at admission, gender, insurance type (one-hot encoded)
- **Admission details**: admission type, discharge disposition (one-hot encoded)

### Model
- **Architecture**: Feedforward neural network (64 → 32 neurons) with BatchNorm, LeakyReLU, and Dropout
- **Loss**: BCEWithLogitsLoss with positive class weighting (handles 80/20 class imbalance)
- **Optimizer**: Adam with learning rate scheduling and early stopping
- **Target**: `readmitted_30d` — binary classification (readmitted within 30 days)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- PostgreSQL 16+
- Docker Desktop
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/davforson/Hospital-Readmission-Risk-Prediction-System.git
cd Hospital-Readmission-Risk-Prediction-System
```

### 2. Set Up the Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 4. Set Up the Database
```bash
python scripts/setup_database.py
```

### 5. Run the Pipeline
```bash
# Extract data
python -c "
from src.extraction.db_extractor import DatabaseExtractor
from src.extraction.csv_extractor import CSVExtractor
db = DatabaseExtractor()
csv_ext = CSVExtractor()
db.extract_patients().to_parquet('data/raw/patients.parquet', index=False)
db.extract_admissions().to_parquet('data/raw/admissions.parquet', index=False)
csv_ext.extract_lab_results().to_parquet('data/raw/lab_results.parquet', index=False)
"

# Clean and build features
python -c "
import pandas as pd
from src.transformation.clean import clean_patients, clean_admissions, clean_lab_results
from src.transformation.features import build_all_features
patients = clean_patients(pd.read_parquet('data/raw/patients.parquet'))
admissions = clean_admissions(pd.read_parquet('data/raw/admissions.parquet'))
labs = clean_lab_results(pd.read_parquet('data/raw/lab_results.parquet'))
build_all_features(patients, admissions, labs)
"

# Train the model
python -c "
from src.model.train import train_model
train_model()
"
```

### 6. Start the Services
```bash
# Terminal 1: Start the prediction API
python -m uvicorn api.main:app --port 8000

# Terminal 2: Start the clinician prediction interface
streamlit run dashboard/predict.py

# Terminal 3: Start the monitoring dashboard
streamlit run dashboard/app.py --server.port 8502
```

### 7. Using Docker (All Services)
```bash
docker-compose up -d

# Access points:
# API:        http://localhost:8000/docs
# Airflow:    http://localhost:8080
# MLflow:     http://localhost:5000
# Dashboard:  http://localhost:8501
```

---

## 🖥️ Clinician Prediction Interface

The Streamlit prediction interface allows clinicians to assess readmission risk without any technical knowledge:

1. Enter patient demographics, admission details, clinical history, and lab results through an intuitive form
2. Click **"Predict Readmission Risk"**
3. Receive a color-coded risk assessment (🟢 Low / 🟡 Medium / 🔴 High) with specific care recommendations

```bash
streamlit run dashboard/predict.py
```

---

## 📊 Monitoring Dashboard

The monitoring dashboard tracks four categories of model health:

- **Model Performance**: F1 score, accuracy, precision, recall trends
- **Data Drift**: KS test comparing production vs. training feature distributions
- **Prediction Drift**: Shift in predicted probability distributions
- **Data Quality**: Null rates, out-of-range values, volume changes

Includes a drift simulation slider to demonstrate how the dashboard responds when data distributions shift.

```bash
streamlit run dashboard/app.py
```

---

## 🔄 CI/CD Pipeline

Every push triggers automated quality checks via GitHub Actions:

- **Formatting**: `black` enforces consistent code style
- **Linting**: `flake8` catches code quality issues
- **Testing**: `pytest` runs model architecture tests
- **Docker**: Builds the container image on merge to main

---

## 📈 MLflow Experiment Tracking

All training runs are logged with:
- Hyperparameters (learning rate, hidden dims, dropout, batch size)
- Metrics per epoch (train loss, validation loss)
- Final evaluation metrics (F1, AUC-ROC, precision, recall)
- Model artifacts (model weights, scaler, feature list)
- Model Registry with staging/production versioning

```bash
mlflow ui
# Open http://localhost:5000
```

---

## 🗓️ Pipeline Orchestration (Airflow)

Two DAGs automate the daily workflow:

**ETL DAG** (6:00 AM daily):
`extract_all_sources` → `validate_data_quality` → `transform_and_engineer`

**Training DAG** (8:00 AM daily):
`wait_for_etl` → `train_model` → `evaluate_model` → `check_performance` → `register_model` / `skip_registration`

---

## 🧪 Data Validation

Great Expectations enforces data contracts before any transformation:

- **Patients**: unique IDs, valid gender/insurance values, no future birth dates
- **Admissions**: valid admission types, minimum 1-day stay, non-null target variable
- **Lab Results**: known test names, non-negative values (with 1% tolerance for noise)

---

## 📬 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit patient features, receive readmission probability |
| `/health` | GET | Check API status and model availability |
| `/docs` | GET | Interactive Swagger API documentation |

---

## 🛣️ Future Enhancements

- [ ] **Apache Kafka** — Real-time patient event streaming for live risk updates
- [ ] **Kubernetes** — Container orchestration for horizontal scaling
- [ ] **Streamlit Cloud** — Deploy the monitoring dashboard publicly
- [ ] **Feature Store** — Centralized feature management with Feast
- [ ] **A/B Testing** — Compare model versions in production

---

## 👤 Author

**David Forson**

Built as a comprehensive MLOps portfolio project demonstrating production-grade machine learning engineering skills for healthcare data science roles.
