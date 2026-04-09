# scripts/setup_database.py

"""
Creates tables and loads synthetic patient data into PostgreSQL.
Run this ONCE before running the extraction pipeline.

Usage:
    python scripts/setup_database.py
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()


def get_engine():
    return create_engine(
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )


def create_tables(engine):
    """Create the hospital database schema."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id VARCHAR(10) PRIMARY KEY,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                date_of_birth DATE,
                gender VARCHAR(10),
                race VARCHAR(30),
                zip_code VARCHAR(10),
                insurance_type VARCHAR(30),
                primary_care_physician VARCHAR(100)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS admissions (
                admission_id VARCHAR(12) PRIMARY KEY,
                patient_id VARCHAR(10) REFERENCES patients(patient_id),
                admission_date TIMESTAMP,
                discharge_date TIMESTAMP,
                admission_type VARCHAR(20),
                discharge_disposition VARCHAR(30),
                primary_diagnosis_code VARCHAR(10),
                primary_diagnosis_desc VARCHAR(200),
                number_of_procedures INTEGER,
                number_of_diagnoses INTEGER,
                length_of_stay INTEGER,
                readmitted_30d BOOLEAN
            );
        """))
        conn.commit()
    print("Tables created successfully.")


def generate_patients(n=5000):
    """Generate synthetic patient demographics."""
    np.random.seed(42)

    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer",
                   "Michael", "Linda", "David", "Elizabeth", "William", "Barbara",
                   "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah",
                   "Christopher", "Karen", "Daniel", "Lisa", "Matthew", "Nancy"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
                  "Miller", "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor",
                  "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson",
                  "White", "Harris", "Clark", "Lewis", "Robinson", "Walker"]
    genders = ["Male", "Female", "Other"]
    races = ["White", "Black", "Hispanic", "Asian", "Other"]
    insurance_types = ["Medicare", "Medicaid", "Private", "Self-Pay", "VA"]

    patients = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n + 1)],
        "first_name": np.random.choice(first_names, n),
        "last_name": np.random.choice(last_names, n),
        "date_of_birth": [
            datetime(1940, 1, 1) + timedelta(days=int(d))
            for d in np.random.uniform(0, 25000, n)
        ],
        "gender": np.random.choice(genders, n, p=[0.48, 0.48, 0.04]),
        "race": np.random.choice(races, n, p=[0.6, 0.13, 0.18, 0.06, 0.03]),
        "zip_code": [f"{z:05d}" for z in np.random.randint(10000, 99999, n)],
        "insurance_type": np.random.choice(insurance_types, n, p=[0.35, 0.2, 0.3, 0.1, 0.05]),
        "primary_care_physician": [f"Dr. {np.random.choice(last_names)}" for _ in range(n)],
    })
    return patients


def generate_admissions(patients_df, avg_admissions_per_patient=2.5):
    """Generate synthetic admission records with realistic readmission patterns."""
    np.random.seed(42)
    records = []
    admission_counter = 1

    icd_codes = {
        "I50.9": "Heart failure, unspecified",
        "J44.1": "COPD with acute exacerbation",
        "E11.9": "Type 2 diabetes without complications",
        "N17.9": "Acute kidney failure, unspecified",
        "I21.9": "Acute myocardial infarction, unspecified",
        "J18.9": "Pneumonia, unspecified organism",
        "K92.2": "Gastrointestinal hemorrhage, unspecified",
        "I63.9": "Cerebral infarction, unspecified",
        "A41.9": "Sepsis, unspecified organism",
        "J96.0": "Acute respiratory failure",
    }

    admission_types = ["Emergency", "Urgent", "Elective", "Trauma"]
    discharge_dispositions = ["Home", "SNF", "Rehab", "Home Health", "AMA", "Expired"]

    for _, patient in patients_df.iterrows():
        n_admissions = max(1, int(np.random.exponential(avg_admissions_per_patient)))
        n_admissions = min(n_admissions, 8)

        base_date = datetime(2022, 1, 1) + timedelta(days=int(np.random.uniform(0, 365)))

        for i in range(n_admissions):
            admission_date = base_date + timedelta(days=int(np.random.uniform(0, 730)))
            los = max(1, int(np.random.exponential(5)))
            discharge_date = admission_date + timedelta(days=los)

            diag_code = np.random.choice(list(icd_codes.keys()))

            # Readmission probability based on realistic risk factors
            base_readmit_prob = 0.15
            if patient["insurance_type"] == "Medicaid":
                base_readmit_prob += 0.05
            if diag_code in ["I50.9", "J44.1", "A41.9"]:
                base_readmit_prob += 0.08
            if los > 7:
                base_readmit_prob += 0.06

            readmitted = np.random.random() < base_readmit_prob

            records.append({
                "admission_id": f"A{str(admission_counter).zfill(8)}",
                "patient_id": patient["patient_id"],
                "admission_date": admission_date,
                "discharge_date": discharge_date,
                "admission_type": np.random.choice(
                    admission_types, p=[0.45, 0.25, 0.2, 0.1]
                ),
                "discharge_disposition": np.random.choice(
                    discharge_dispositions, p=[0.45, 0.2, 0.1, 0.15, 0.05, 0.05]
                ),
                "primary_diagnosis_code": diag_code,
                "primary_diagnosis_desc": icd_codes[diag_code],
                "number_of_procedures": np.random.randint(0, 6),
                "number_of_diagnoses": np.random.randint(1, 12),
                "length_of_stay": los,
                "readmitted_30d": readmitted,
            })
            admission_counter += 1

    return pd.DataFrame(records)


def generate_lab_results_csv(patients_df, output_path="data/lab_results.csv"):
    """Generate synthetic lab results and save as CSV."""
    np.random.seed(42)
    records = []

    lab_tests = {
        "glucose": {"mean": 140, "std": 50, "unit": "mg/dL"},
        "creatinine": {"mean": 1.2, "std": 0.6, "unit": "mg/dL"},
        "hemoglobin": {"mean": 13.5, "std": 2.0, "unit": "g/dL"},
        "white_blood_cell": {"mean": 8.0, "std": 3.0, "unit": "K/uL"},
        "sodium": {"mean": 140, "std": 4, "unit": "mEq/L"},
        "potassium": {"mean": 4.2, "std": 0.6, "unit": "mEq/L"},
        "bun": {"mean": 18, "std": 8, "unit": "mg/dL"},
        "hba1c": {"mean": 6.5, "std": 1.5, "unit": "%"},
    }

    sample_patients = patients_df.sample(n=min(3000, len(patients_df)), random_state=42)

    for _, patient in sample_patients.iterrows():
        n_tests = np.random.randint(3, 15)
        for _ in range(n_tests):
            test_name = np.random.choice(list(lab_tests.keys()))
            test_info = lab_tests[test_name]
            records.append({
                "patient_id": patient["patient_id"],
                "test_name": test_name,
                "test_value": round(np.random.normal(test_info["mean"], test_info["std"]), 2),
                "test_unit": test_info["unit"],
                "test_date": datetime(2022, 1, 1) + timedelta(days=int(np.random.uniform(0, 1095))),
                "ordering_physician": f"Dr. {np.random.choice(['Smith', 'Patel', 'Kim', 'Garcia', 'Chen'])}",
            })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Lab results CSV created: {len(df)} records saved to {output_path}")
    return df


def load_to_database(engine, patients_df, admissions_df):
    """Load dataframes into PostgreSQL."""
    patients_df.to_sql("patients", engine, if_exists="replace", index=False)
    print(f"Loaded {len(patients_df)} patients to database.")

    admissions_df.to_sql("admissions", engine, if_exists="replace", index=False)
    print(f"Loaded {len(admissions_df)} admissions to database.")


if __name__ == "__main__":
    print("Setting up hospital readmissions database...")
    print("=" * 50)

    engine = get_engine()

    # Create tables
    create_tables(engine)

    # Generate synthetic data
    print("\nGenerating patient data...")
    patients = generate_patients(n=5000)

    print("Generating admission records...")
    admissions = generate_admissions(patients)

    print("Generating lab results CSV...")
    lab_results = generate_lab_results_csv(patients)

    # Load to database
    print("\nLoading data to PostgreSQL...")
    load_to_database(engine, patients, admissions)

    # Summary
    print("\n" + "=" * 50)
    print("SETUP COMPLETE")
    print(f"  Patients:    {len(patients):,}")
    print(f"  Admissions:  {len(admissions):,}")
    print(f"  Lab Results: {len(lab_results):,}")
    print(f"  Readmission Rate: {admissions['readmitted_30d'].mean():.1%}")
    print("=" * 50)