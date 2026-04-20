import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
import logging

logger = logging.getLogger(__name__)


def prepare_data(feature_path: str = "data/processed/features.parquet"):
    """Load features and return train, val, test splits together with
    scaler, and feature columns.

    """
    df = pd.read_parquet(feature_path)

    target_col = "readmitted_30d"
    feature_cols = [c for c in df.columns if c != target_col]

    # Check for the presence of a column
    if "primary_diagnosis_code" in df.columns:
        df = pd.get_dummies(
            data=df, columns=["primary_diagnosis_code"], prefix=["diag"], drop_first=True
        )
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Split the dataset in train, val and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, stratify=y, test_size=0.3, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Features: {X_train.shape[1]}")
    logger.info(
        f"Readmission rate - Train: {y_train.mean()}, Val: {y_val.mean()}, Test: {y_test.mean()}"
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols


def train_model():
    """Train the readmission prediction model."""
    from src.model.architecture import ReadmissionPredictor

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_cols = prepare_data()

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))

    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)

    # Handle class imbalance with weighted class
    pos_weight = torch.tensor([(1 - y_train.mean()) / y_train.mean()])
    logger.info(f"Class weight for positive(readmitted): {pos_weight.item():.2f}")

    mlflow.set_experiment("readmission_prediction")

    with mlflow.start_run(run_name="pytorch_baseline"):
        # Instantiate class
        model = ReadmissionPredictor(
            input_dim=X_train.shape[1], hidden_dims=[64, 32], dropout_rate=0.2
        )

        # Train loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )

        params = {
            "model_type": "ReadmissionPredictor",
            "batch_size": 256,
            "max_epochs": 200,
            "hidden_dim": "[64-32]",
            "dropout_rate": 0.2,
            "learning_rate": 0.005,
            "weight_decay": 1e-4,
            "patience": 20,
            "factor": 0.5,
            "num_features": X_train.shape[1],
            "train_sample": len(X_train),
        }

        mlflow.log_params(params)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20
        best_model_state = None

        for epoch in range(200):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    predictions = model(X_batch)
                    loss = criterion(predictions, y_batch)

                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            # Log every 10 epochs
            if epoch % 10 == 0:
                logger.info(f"Epoch = {epoch}, train_loss: {train_loss:.4f}, Val_loss: {val_loss}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        from src.model.evaluate import evaluate_model

        # Load the best model state
        model.load_state_dict(best_model_state)

        mlflow.pytorch.log_model(model, "readmission_model")

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_metric("stopped_at_epoch", epoch)

        all_metrics = {
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_f1_score": metrics["f1_score"],
            "test_auc_roc": metrics["auc_roc"],
            "test_recall": metrics["recall"],
        }

        mlflow.log_metrics(all_metrics)

        # Save model and scaler
        torch.save(scaler, "data/processed/scaler.pt")
        mlflow.log_artifact("data/processed/scaler.pt")

        logger.info("Training complete. Model saved.")

        # Log feature list
        import json

        with open("data/processed/feature_cols.json", "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact("data/processed/feature_cols.json")

        logger.info(f"MLflow run complete. F1: {metrics['f1_score']:.4f}")

        return model, scaler, (X_test, y_test), feature_cols
