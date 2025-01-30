import os
from datetime import datetime
from typing import Tuple

import dvc.api
import mlflow
import polars as pl
from loguru import logger

from ml_project.data.data_loader import DataManager
from ml_project.features.preprocessing import (
    LagFeatureGenerator,
    build_preprocessor,
)
from ml_project.mlflow.tracking import MLflowTracker
from ml_project.models.train import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        self.data_manager = DataManager()
        self.tracker = MLflowTracker()

    def run(self):
        self.tracker.setup()
        with self.tracker.start_run(f"training_pipeline_{datetime.now()}"):
            try:
                # Data loading and preprocessing
                raw_data = self.data_manager.load_raw_data()
                processed_data = self._preprocess_data(raw_data)

                # Feature engineering
                X, y = self._prepare_features(processed_data)

                # Model training
                preprocessor = build_preprocessor()
                X_processed = preprocessor.fit_transform(X)

                trainer = ModelTrainer(preprocessor)
                metrics = trainer.train(X_processed, y)

                # Log data version
                mlflow.log_param(
                    "data_version", dvc.api.get_url(path=os.getenv("DATA_PATH"))
                )
                mlflow.log_artifact(os.getenv("DATA_PATH"))

                logger.success("Pipeline completed successfully")
                return metrics

            except Exception as e:
                logger.error(f"Pipeline failed: {str(e)}")
                raise

    def _preprocess_data(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.sort(["Year", "Month"])
        df = LagFeatureGenerator(lag_periods=3).fit_transform(df)
        return df

    def _prepare_features(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.Series]:
        X = df.drop(["Quantity"])
        y = df["Quantity"]
        return X, y
