import os
from typing import Tuple

import polars as pl
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing utilizing Polars.
    """

    def __init__(self) -> None:
        """
        Initializes the DataPreprocessor. Variables for encoders can be stored here.
        """
        self.encoders: dict[str, LabelEncoder] = {}

    def _find_csv_file(self, directory: str, dataset_prefix: str) -> str:
        """
        Locates a candidate CSV file within a given dataset directory.
        Uses the provided prefix, e.g., 'HI-Small' to find the exact transaction file.
        """
        expected_file = f"{dataset_prefix}_Trans.csv"
        path = os.path.join(directory, expected_file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected file {expected_file} not found in {directory}"
            )
        return path

    def load_data(
        self, dataset_path: str, dataset_prefix: str = "HI-Small"
    ) -> pl.DataFrame:
        """
        Loads the dataset into a Polars DataFrame.
        """
        if os.path.isdir(dataset_path):
            csv_path = self._find_csv_file(dataset_path, dataset_prefix)
        else:
            csv_path = dataset_path

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset path does not exist: {csv_path}")

        df = pl.read_csv(csv_path)
        return df

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Performs basic cleaning: handling missing values by filling or dropping.
        Drops unnecessary raw formatting issues if any.
        """
        # For IBM AML, basic handling of nulls if present
        df_clean = df.drop_nulls()
        return df_clean

    def encode_features(
        self, df: pl.DataFrame, categorical_cols: list[str]
    ) -> pl.DataFrame:
        """
        Encodes string categorical columns to numeric using LabelEncoder.
        Saves the encoders for future inverse transforms if necessary.
        """
        encoded_dict = {}

        for col in df.columns:
            if col in categorical_cols:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()

                # Convert to pandas series for LabelEncoder
                series = df[col].to_pandas()
                encoded_vals = self.encoders[col].fit_transform(series)
                encoded_dict[col] = pl.Series(col, encoded_vals)
            else:
                encoded_dict[col] = df[col]

        return pl.DataFrame(encoded_dict)

    def split_data(
        self, df: pl.DataFrame, target_col: str
    ) -> Tuple[
        pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.Series, pl.Series, pl.Series
    ]:
        """
        Splits data into train, validate, and test sets (60% / 20% / 20%).
        Returns X_train, X_val, X_test, y_train, y_val, y_test.
        """
        y = df[target_col]
        X = df.drop(target_col)

        # First split off 40% for validation and testing (remaining 60% for training)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X.to_pandas(), y.to_pandas(), test_size=0.4, random_state=42
        )

        # Split the 40% evenly into 20% validate and 20% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        return (
            pl.from_pandas(X_train),
            pl.from_pandas(X_val),
            pl.from_pandas(X_test),
            pl.Series(y_train),
            pl.Series(y_val),
            pl.Series(y_test),
        )
