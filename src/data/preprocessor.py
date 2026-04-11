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

    def _find_csv_file(self, directory: str) -> str:
        """
        Locates a candidate CSV file within a given dataset directory.
        Prioritizes smaller dataset for development if available.
        """
        files = os.listdir(directory)
        csv_files = [f for f in files if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {directory}")

        for pref in ["HI-Small_Trans.csv", "HI-Medium_Trans.csv"]:
            if pref in csv_files:
                return os.path.join(directory, pref)
        return os.path.join(directory, csv_files[0])

    def load_data(self, dataset_path: str) -> pl.DataFrame:
        """
        Loads the dataset into a Polars DataFrame.
        """
        if os.path.isdir(dataset_path):
            csv_path = self._find_csv_file(dataset_path)
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
        self, df: pl.DataFrame, target_col: str, test_size: float = 0.2
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
        """
        Splits data into train and test sets for ML validation.
        Returns X_train, X_test, y_train, y_test.
        """
        y = df[target_col]
        X = df.drop(target_col)

        X_train, X_test, y_train, y_test = train_test_split(
            X.to_pandas(), y.to_pandas(), test_size=test_size, random_state=42
        )

        return (
            pl.from_pandas(X_train),
            pl.from_pandas(X_test),
            pl.Series(y_train),
            pl.Series(y_test),
        )
