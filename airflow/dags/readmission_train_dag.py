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
    from src.model.train import train_model as run_training 

    model, scaler, (X_test, y_test), feature_cols = run_training()

    context["ti"].xcom_push(key="training_complete", value=True)
    print("Model training complete")
    

def evaluate_model(**context):
    """Evaluate model performance"""
    import torch 
    from src.model.architecture import ReadmissionPredictor
    from src.model.evaluate import evaluate_model as run_evaluation
    import json

    #Load the trained model
    with open("data/processed/features.json", "r") as f:
        feature_cols = json.load(f)

    model = ReadmissionPredictor(input_dim=len(feature_cols), hidden_dims=[64,32], dropout_rate=0.2)
    model.load_state_dict(torch.load("data/processed/model.pt", weights_only=True))

    #Load test data
    scaler = torch.load("data/processed/scaler.pt", weights_only=False)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet("data/processed/features.parquet")
    target_col = "readmitted_30d"

    feature_cols_all = [c for c in df.columns if c != target_col]

    if "primary_diagnosis_code" in df.columns:
        df = pd.get_dummies(df, columns=["primary_diagnosis_code"], prefix=["diag"], drop_first=True)
        feature_cols_all = [c for c in df.columns if c != target_col]

    X = df[feature_cols_all].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    X_test = scaler.transform(X_test)

    metrics = run_evaluation(model, X_test, y_test)

    context["ti"].xcom_push(key="f1_score", value = metrics["f1_score"])
    context["ti"].xcom_push(key="auc_roc", value = metrics["auc_roc"])
    print(f"Model F1:{metrics["f1_score"]:.4f}, AUC: {metrics["auc_roc"]:.4f}")

def check_model_performance(**context):
    """Branch: register model only if performance meets threshold"""
    f1 = context["ti"].xcom_pull(task_ids='evaluate_model', key="f1_score")
    if f1 > 0.70:
        return "register_model"
    else:
        return "skip_registration"
    
def register_model(**context):
    """Register model in MLflow Registry"""
    import mlflow
    from src.model.registry import register_model_if_qualified

    mlflow.set_experiment("readmission_prediction")
    runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
    run_id = runs.iloc[0].run_id

    version = register_model_if_qualified(run_id, min_f1=0.30)
    if version:
        print(f"Model was registered as version {version}")
    else:
        print(f"Model did not meet threshold")

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
        mode = "reschedule"
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