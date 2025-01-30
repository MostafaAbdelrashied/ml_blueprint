import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, lag_periods=3):
        self.lag_periods = lag_periods

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame):
        X = X.clone()
        for lag in range(1, self.lag_periods + 1):
            X = X.with_columns(
                pl.col("Quantity").shift(-lag).alias(f"Quantity_Lag_{lag}")
            )
        return X.drop_nulls()


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "Year",
        "Month",
        "Quantity_Lag_1",
        "Quantity_Lag_2",
        "Quantity_Lag_3",
    ]
    categorical_features = ["Make"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor
