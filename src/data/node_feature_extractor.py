"""Module to extract structural and financial node features from AML data."""

import polars as pl
import torch


class NodeFeatureExtractor:
    """Computes node features from accounts and transactions."""

    def __init__(self) -> None:
        """Initializes the node feature extractor."""

    def compute_features(
        self,
        accounts_df: pl.DataFrame,
        trans_df: pl.DataFrame,
        train_cutoff: int,
    ) -> torch.Tensor:
        """Computes structural and financial statistics for nodes."""
        train_trans = trans_df.slice(0, train_cutoff)
        out_agg = self._aggregate_outgoing(train_trans)
        in_agg = self._aggregate_incoming(train_trans)
        node_df = accounts_df.select(["Account_ID"])
        node_df = (
            node_df.join(out_agg, left_on="Account_ID", right_on="From_Acc", how="left")
            .join(in_agg, left_on="Account_ID", right_on="To_Acc", how="left")
            .fill_null(0.0)
        )
        node_df = self._apply_log1p(node_df)
        feature_cols = [
            "out_degree",
            "total_sent",
            "avg_sent",
            "max_sent",
            "min_sent",
            "in_degree",
            "total_received",
            "avg_received",
            "max_received",
            "min_received",
        ]
        x_tensor = torch.tensor(node_df.select(feature_cols).to_numpy(), dtype=torch.float)
        return self._scale_tensor(x_tensor)

    def _aggregate_outgoing(self, train_trans: pl.DataFrame) -> pl.DataFrame:
        """Aggregates outgoing transactions statistics."""
        return train_trans.group_by("From_Acc").agg(
            [
                pl.len().alias("out_degree"),
                pl.col("Amount Paid").sum().alias("total_sent"),
                pl.col("Amount Paid").mean().alias("avg_sent"),
                pl.col("Amount Paid").max().alias("max_sent"),
                pl.col("Amount Paid").min().alias("min_sent"),
            ]
        )

    def _aggregate_incoming(self, train_trans: pl.DataFrame) -> pl.DataFrame:
        """Aggregates incoming transactions statistics."""
        return train_trans.group_by("To_Acc").agg(
            [
                pl.len().alias("in_degree"),
                pl.col("Amount Received").sum().alias("total_received"),
                pl.col("Amount Received").mean().alias("avg_received"),
                pl.col("Amount Received").max().alias("max_received"),
                pl.col("Amount Received").min().alias("min_received"),
            ]
        )

    def _apply_log1p(self, node_df: pl.DataFrame) -> pl.DataFrame:
        """Applies log1p transformation to financial columns."""
        financial_cols = [
            "total_sent",
            "avg_sent",
            "max_sent",
            "min_sent",
            "total_received",
            "avg_received",
            "max_received",
            "min_received",
        ]
        return node_df.with_columns([pl.col(c).log1p() for c in financial_cols])

    def _scale_tensor(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes a feature tensor using PyTorch-native scaling."""
        mean: torch.Tensor = feature_tensor.mean(dim=0, keepdim=True)
        std: torch.Tensor = feature_tensor.std(dim=0, correction=0, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (feature_tensor - mean) / std
