from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import timedelta
import logging

default_args = {
    "owner": "David",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": True,
    "email": ["ml-alerts@hospital.com"]
}


def train_model(**context):
    """Train the readmission prediction model"""
    # We'll fill this in section 6

    import pandas as pd
    features = pd.read_parquet("data/processed/features.parquet")
    context["ti"].xcom_push(key="training_rows", value=len(features))
    print(f"Training on {len(features)} rows")

def evaluate_model(**context):
    """Evaluate model performance"""
    # Placeholder - fill this in section 6

    f1_score = 0.75
    context["ti"].xcom_push(key="f1_score", value=f1_score)
    print(f"Model f1 score: {f1_score}")

def check_model_performance(**context):
    """Branch: register model only if performance meets threshold"""
    f1 = context["ti"].xcom_pull(task_ids='evaluate_model', key="f1_score")
    if f1 > 0.70:
        return "register_model"
    else:
        return "skip_registration"
    
def register_model(**context):
    """Register model in MLflow Registry"""
    print("Model registered")

def skip_registration(**context):
    """Log models that didn't meet threshold """
    f1 = context["ti"].xcom_pull(task_ids="evaluate_model", key="f1_score")
    print(f"Model F1 {f1} below threshold. Skipping registration")


with DAG(
    dag_id = "readmission_training_pipeline",
    default_args = default_args,
    description = "Training, evaluating and registering readmission prediction models",
    schedule_interval = "0 8 * * *",
    start_date = days_ago(1),
    catchup = False,
    tags = ["readmission", "training"]
) as dag:
    
    # Wait for etl pipeline to complete
    wait_for_etl = ExternalTaskSensor(
        task_id = "wait_for_etl",
        external_dag_id = "readmission_etl_pipeline",
        external_task_id = "transform_and_engineer",
        timeout = 3600,
        poke_interval = 60,
        mode = "poke"
    )

    train = PythonOperator(
        task_id = "train_model",
        python_callable = train_model
    )

    evaluate = PythonOperator(
        task_id = "evaluate_model",
        python_callable = evaluate_model
    )

    check_perf = BranchPythonOperator(
        task_id = "check_performance",
        python_callable = check_model_performance
    )

    register = PythonOperator(
        task_id = "register_model",
        python_callable = register_model
    )

    skip = PythonOperator(
        task_id = "skip_registration",
        python_callable = skip_registration
    )


    # Task dependencies
    wait_for_etl >> train >> evaluate >> check_perf
    check_perf >> [register, skip]