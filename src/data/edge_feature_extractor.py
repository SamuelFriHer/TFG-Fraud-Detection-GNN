"""Module to extract and encode edge features from AML transactional data."""

import polars as pl
import torch


class EdgeFeatureExtractor:
    """Extracts, encodes, and scales edge features."""

    def __init__(self) -> None:
        """Initializes the edge feature extractor."""
        self.edge_encoders: dict[str, list[str]] = {}

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
        expressions = []
        for col in df.columns:
            if col in cols:
                if col not in self.edge_encoders:
                    self.edge_encoders[col] = df[col].unique().drop_nulls().sort().to_list()
                categories = self.edge_encoders[col]
                expressions.append(
                    pl.col(col)
                    .cast(pl.Enum(categories))
                    .to_physical()
                    .cast(pl.Int64)
                    .cast(pl.Float32)
                )
            else:
                if df.schema[col] == pl.String:
                    if col == "Timestamp":
                        expressions.append(pl.col(col).cast(pl.Float32))
                    else:
                        raise ValueError(f"Unhandled string column: {col}")
                else:
                    expressions.append(pl.col(col).cast(pl.Float32))

        encoded_df = df.select(expressions)
        return torch.tensor(encoded_df.to_numpy(), dtype=torch.float)

    def _scale_tensor(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes a feature tensor using PyTorch-native scaling."""
        mean: torch.Tensor = feature_tensor.mean(dim=0, keepdim=True)
        std: torch.Tensor = feature_tensor.std(dim=0, correction=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (feature_tensor - mean) / std
