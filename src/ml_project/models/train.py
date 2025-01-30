import mlflow
import numpy as np
import polars as pl
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score


class ModelTrainer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.model = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=42
        )

    def train(self, X: pl.DataFrame, y: pl.Series):
        # Log parameters
        mlflow.log_params(self.model.get_params())

        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(
            self.model, X, y, cv=tscv, scoring="neg_mean_absolute_error"
        )

        # Train final model
        self.model.fit(X, y)

        # Log metrics
        metrics = {
            "mae_cv": np.mean(np.abs(scores)),
            "rmse_cv": np.sqrt(np.mean(np.abs(scores))),
        }
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="model",
            input_example=X,
            registered_model_name="rf_model",
        )

        logger.success("Training completed")
        return metrics
