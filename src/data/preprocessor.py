"""Data loading, cleaning, encoding, and splitting utilities using Polars."""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Handles data loading, cleaning, and preprocessing utilizing Polars."""

    def __init__(self) -> None:
        """Initializes the DataPreprocessor with an empty encoder registry."""
        self.encoders: dict[str, list[str]] = {}

    def _find_csv_file(self, directory: str, dataset_prefix: str) -> str:
        """Locates a CSV file in a directory using the given dataset prefix."""
        expected_file = f"{dataset_prefix}_Trans.csv"
        path = Path(directory) / expected_file
        if not path.exists():
            raise FileNotFoundError(f"Expected file {expected_file} not found in {directory}")
        return str(path)

    def load_data(self, dataset_path: str, dataset_prefix: str = "HI-Small") -> pl.DataFrame:
        """Loads the dataset CSV into a Polars DataFrame."""
        resolved_path = Path(dataset_path)

        if resolved_path.is_dir():
            csv_path = Path(self._find_csv_file(dataset_path, dataset_prefix))
        else:
            csv_path = resolved_path

        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")

        df = pl.read_csv(str(csv_path))
        # Handle duplicate "Account" columns renamed by Polars
        if "Account_duplicated_0" in df.columns:
            df = df.rename({"Account_duplicated_0": "Account.1"})
        return df

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drops rows with null values as basic cleaning for the IBM AML dataset."""
        return df.drop_nulls()

    def encode_features(self, df: pl.DataFrame, categorical_cols: list[str]) -> pl.DataFrame:
        """Label-encodes string categorical columns, saving categories for consistency."""
        if not categorical_cols:
            return df

        cols_to_encode: list[str] = [col for col in categorical_cols if col not in self.encoders]
        if cols_to_encode:
            unique_lists: pl.DataFrame = df.select(
                [pl.col(col).unique().drop_nulls().sort().implode() for col in cols_to_encode]
            )
            categories_tuple: tuple[list[str], ...] = unique_lists.row(0)
            for col, categories in zip(cols_to_encode, categories_tuple):
                self.encoders[col] = categories

        expressions = [
            pl.col(col).cast(pl.Enum(self.encoders[col])).to_physical().cast(pl.Int64)
            for col in categorical_cols
        ]
        return df.with_columns(expressions)

    def split_data(
        self,
        df: pl.DataFrame,
        target_col: str,
        test_size: float = 0.4,
        random_state: int = 42,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Splits data into train / validate / test sets (60% / 20% / 20%).

        Returns numpy arrays to ensure compatibility across all sklearn-based models,
        avoiding LightGBM's known incompatibility with pandas feature name tracking.
        """
        y_labels = df[target_col].to_numpy()
        x_features = df.drop(target_col).to_numpy()

        x_train, x_temp, y_train, y_temp = train_test_split(
            x_features, y_labels, test_size=test_size, random_state=random_state
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=random_state
        )
        return x_train, x_val, x_test, y_train, y_val, y_test
