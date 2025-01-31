import json
import os
from pathlib import Path

import dvc.api
import polars as pl
from loguru import logger


class DataManager:
    def __init__(self, data_path: Path = None):
        self.data_path = data_path or os.getenv("DATA_PATH")

    def load_raw_data(self) -> pl.DataFrame:
        try:
            with dvc.api.open(str(self.data_path), repo=".", mode="r") as fd:
                df = pl.read_csv(fd)
                logger.info(f"Loaded raw data with {len(df)} records")
                return df
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            raise

    def save_processed_data(self, df: pl.DataFrame, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path)
            logger.success(f"Saved processed data to {path}")
        except Exception as e:
            logger.error(f"Data saving failed: {str(e)}")
            raise

    def calculate_stats(self, df: pl.DataFrame, is_store: bool = False) -> dict:
        # generate some stats about the data
        stats_dict = {
            "size": df.shape[0],
            "num_features": df.shape[1],
            "columns": df.columns,
            "null_counts": df.null_count().sum_horizontal()[0],
            "Quantity_mean": df["Quantity"].mean(),
            "Quantity_std": df["Quantity"].std(),
        }
        if is_store:
            with open("reports/data/data_stats.json", "w") as f:
                json.dump(stats_dict, f)
        return stats_dict


if __name__ == "__main__":
    dl = DataManager()
    df = dl.load_raw_data()
    dl.save_processed_data(df, Path("data/processed/cleaned.parquet"))
    dl.calculate_stats(df, is_store=True)
