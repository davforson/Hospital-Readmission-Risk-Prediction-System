import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """Evaluate model on test set with multiple metrics.

    Args:
        model: trained PyTorch model
        X_test: scaled test features (numpy array)
        y_test: test labels (numpy array)
        threshold: probability threshold for classification (default 0.5)

    Returns:
        dict of metrics
    """
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        probabilities = torch.sigmoid(model(X_tensor)).numpy()

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "auc_roc": roc_auc_score(y_test, probabilities),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    metrics["true_negatives"] = int(cm[0][0])
    metrics["false_positives"] = int(cm[0][1])
    metrics["false_negatives"] = int(cm[1][0])
    metrics["true_positives"] = int(cm[1][1])

    # Log results
    logger.info("=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"  TN={metrics['true_negatives']}  FP={metrics['false_positives']}")
    logger.info(f"  FN={metrics['false_negatives']}  TP={metrics['true_positives']}")
    logger.info("=" * 50)

    return metrics
