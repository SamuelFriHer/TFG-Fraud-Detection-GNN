"""Module to extract and encode edge features from AML transactional data."""

import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler


class EdgeFeatureExtractor:
    """Extracts, encodes, and scales edge features."""

    def __init__(self) -> None:
        """Initializes the edge feature extractor."""
        self.edge_encoders: dict[str, LabelEncoder] = {}

    def extract_features(self, trans_df: pl.DataFrame) -> torch.Tensor:
        """Extracts, encodes, and scales edge features from transactions."""
        trans_df = trans_df.with_columns(
            [pl.col("Amount Received").log1p(), pl.col("Amount Paid").log1p()]
        )
        edge_cols = ["Receiving Currency", "Payment Currency", "Payment Format"]
        drop_cols = [
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Is Laundering",
            "From_Acc",
            "To_Acc",
        ]
        edge_features_df = trans_df.drop(drop_cols)
        edge_attr = self._encode_dataframe(edge_features_df, edge_cols)
        return self._scale_tensor(edge_attr)

    def _encode_dataframe(self, df: pl.DataFrame, cols: list[str]) -> torch.Tensor:
        """Encodes DataFrame columns to a PyTorch float tensor."""
        encoded_dict: dict[str, pl.Series] = {}
        for col in df.columns:
            if col in cols:
                encoded_dict[col] = self._encode_categorical_col(df, col)
            else:
                encoded_dict[col] = self._handle_numerical_col(df, col)

        encoded_df = pl.DataFrame(encoded_dict)
        return torch.tensor(encoded_df.to_numpy(), dtype=torch.float)

    def _encode_categorical_col(self, df: pl.DataFrame, col: str) -> pl.Series:
        """Encodes a categorical column using a LabelEncoder."""
        if col not in self.edge_encoders:
            self.edge_encoders[col] = LabelEncoder()
        pandas_series = df[col].to_pandas()
        encoded_vals = self.edge_encoders[col].fit_transform(pandas_series)
        return pl.Series(col, encoded_vals)

    def _handle_numerical_col(self, df: pl.DataFrame, col: str) -> pl.Series:
        """Handles numeric or timestamp columns."""
        if df.schema[col] == pl.String:
            if col == "Timestamp":
                return df[col].cast(pl.Float32)
            raise ValueError(f"Columna string no manejada: {col}")
        return df[col].cast(pl.Float32)

    def _scale_tensor(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes a feature tensor using StandardScaler."""
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(feature_tensor.numpy())
        return torch.tensor(scaled_array, dtype=torch.float)
