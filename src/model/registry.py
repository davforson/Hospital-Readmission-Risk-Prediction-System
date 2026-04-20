import mlflow
from mlflow.tracking import MlflowClient
import logging

logger = logging.getLogger(__name__)


def register_model_if_qualified(run_id: str, min_f1: float = 0.30):
    """Register model only if it meets the performance threshold.

    Model Registry stages:
      None       → just registered, no stage
      Staging    → under evaluation by the team
      Production → actively serving predictions
      Archived   → replaced by a newer version
    """

    client = MlflowClient()
    run = client.get_run(run_id)
    f1 = run.data.metrics.get("test_f1_score", 0)

    if f1 >= min_f1:
        model_uri = f"runs:/{run_id}/readmission_model"
        result = mlflow.register_model(model_uri, "ReadmissionPredictor")

        client.transition_model_version_stage(
            name="ReadmissionPredictor", version=result.version, stage="Staging"
        )
        logger.info(f"Model v{result.version} registerd and moved to Staging. (F1: {f1:.4f})")
        return result.version
    else:
        logger.info(f"Model rejected. F1 {f1:.4f} < threshold {min_f1:.4f}")


def promote_to_production(version: str):
    "Promote a stage model to a production model"
    client = MlflowClient()

    client.transition_model_version_stage(
        name="ReadmissionPredictor", version=version, stage="Production"
    )

    logger.info("Model v{version} promoted from Staging to Production")
