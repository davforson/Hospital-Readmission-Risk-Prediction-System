from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Default arguments applied to all tasks in the DAG
default_args = {
    "owner": "David",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": True,
    "email": ["data-alerts@hospital.com"]
}

# Task function
def run_extraction(**context):
    """Extract data from all sources and save to raw staging"""
    from src.extraction.db_extractor import DatabaseExtractor
    from src.extraction.csv_extractor import CSVExtractor

    db = DatabaseExtractor()
    csv_ext = CSVExtractor()

    patients = db.extract_patients()
    admissions = db.extract_admissions()
    lab_results = csv_ext.extract_lab_results()

    patients.to_parquet("data/raw/patients.parquet", index=False)
    admissions.to_parquet("data/raw/admissions.parquet", index=False)
    lab_results.to_parquet("data/raw/lab_results.parquet", index=False)

    # Push metadat via XCom
    context["ti"].xcom_push(key="patient_count",value=len(patients))
    context["ti"].xcom_push(key="admission_count",value=len(admissions))
    context["ti"].xcom_push(key="lab_count",value=len(lab_results))

    logger.info("Extraction of data complete.")


def run_validation(**context):
    """Validate extracted data. Fails task if any validation fails"""
    import pandas as pd
    from src.validation.expectations import (
        validate_patients, validate_admissions, validate_lab_results
    )

    patients = pd.read_parquet("data/raw/patients.parquet")
    admissions = pd.read_parquet("data/raw/admissions.parquet")
    lab_results = pd.read_parquet("data/lab_results.parquet")

    p_result = validate_patients(patients)
    a_result = validate_admissions(admissions)
    l_result = validate_lab_results(lab_results)

    # If any validation fails, raise an Exception to fail the task
    # Airflow will retry based on the default args, then alert on failure
    if not all([p_result["success"], a_result["success"], l_result["success"]]):
        failed = []
        if not p_result["success"]:
            failed.append("patients")
        if not a_result["success"]:
            failed.append("admissions")
        if not l_result["success"]:
            failed.append("lab_results")
        raise ValueError(f"Data validation failed for: {', '.join(failed)}")
    
    logger.info("All validation passed")


def run_transformation(**context):
    """Clean data and build features"""
    import pandas as pd
    from src.transformation.clean import clean_patients, clean_lab_results, clean_admissions
    from src.transformation.features import build_all_features

    patients = pd.read_parquet("data/raw/patients.parquet")
    admissions = pd.read_parquet("data/raw/admissions.parquet")
    lab_results = pd.read_parquet("data/raw/lab_results.parquet")

    patients_clean = clean_patients(patients)
    admissions_clean = clean_admissions(admissions)
    lab_results_clean = clean_lab_results(lab_results)

    features = build_all_features(patients_clean, admissions_clean, lab_results_clean)

    context["ti"].xcom_push(key="feature_rows", value=len(features))
    context["ti"].xcom_push(key="feature_cols", value=features.shape[1])

    logger.info(f"Transformation complete: {features.shape}")

with DAG(
    dag_id = "readmission_etl_pipeline",
    default_args = default_args,
    description = "Extract, validate and transform patient readmission data",
    schedule_interval = "0 6 * * *",
    start_date = days_ago(1),
    catchup = False,
    tags = ["readmission", "etl"]
) as dag:
    
    extract = PythonOperator(
        task_id = "extract_all_sources",
        python_callable = run_extraction
    )

    validate = PythonOperator(
        task_id = "validate_data_quality",
        python_callable = run_validation
    )

    transform = PythonOperator(
        task_id = "transform_and_engineer",
        python_callable = run_transformation
    )

    # Define task dependencies
    extract >> validate >> transform


