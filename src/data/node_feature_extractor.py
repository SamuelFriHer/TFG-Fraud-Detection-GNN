"""Module to extract structural and financial node features from AML data."""

import numpy as np
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
        neo4j_df: pl.DataFrame | None = None,
    ) -> torch.Tensor:
        """Computes structural and financial statistics for nodes."""
        node_df: pl.DataFrame = self._build_base_node_df(accounts_df, trans_df, train_cutoff)
        feature_cols: list[str] = [
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

        if neo4j_df is not None:
            return self._extract_neo4j_features(node_df, neo4j_df, feature_cols)

        raw_features: np.ndarray = node_df.select(feature_cols).to_numpy()
        x_tensor: torch.Tensor = torch.tensor(raw_features, dtype=torch.float)
        return self._scale_tensor(x_tensor)

    def _build_base_node_df(
        self,
        accounts_df: pl.DataFrame,
        trans_df: pl.DataFrame,
        train_cutoff: int,
    ) -> pl.DataFrame:
        """Builds basic aggregated node DataFrame with financial and degree statistics."""
        train_trans: pl.DataFrame = trans_df.slice(0, train_cutoff)
        out_agg: pl.DataFrame = self._aggregate_outgoing(train_trans)
        in_agg: pl.DataFrame = self._aggregate_incoming(train_trans)
        node_df: pl.DataFrame = accounts_df.select(["Account_ID"])
        node_df = (
            node_df.join(out_agg, left_on="Account_ID", right_on="From_Acc", how="left")
            .join(in_agg, left_on="Account_ID", right_on="To_Acc", how="left")
            .fill_null(0.0)
        )
        return self._apply_log1p(node_df)

    def _extract_neo4j_features(
        self,
        node_df: pl.DataFrame,
        neo4j_df: pl.DataFrame,
        feature_cols: list[str],
    ) -> torch.Tensor:
        """Extracts and scales features from Neo4j integration."""
        joined_df: pl.DataFrame = node_df.join(neo4j_df, on="Account_ID", how="left")

        base_features: np.ndarray = joined_df.select(feature_cols).to_numpy()
        pagerank: np.ndarray = joined_df.select("pagerank").fill_null(0.0).to_numpy()
        continuous_np: np.ndarray = np.hstack([base_features, pagerank])

        continuous_tensor: torch.Tensor = torch.tensor(continuous_np, dtype=torch.float)
        scaled_continuous: torch.Tensor = self._scale_tensor(continuous_tensor)

        wcc_id: np.ndarray = joined_df.select("wcc_id").fill_null(0.0).to_numpy()
        wcc_tensor: torch.Tensor = torch.tensor(wcc_id, dtype=torch.float)

        fastrp_list: list[list[float] | None] = joined_df["fastrp_emb"].to_list()
        dim: int = 64
        fastrp_arr: np.ndarray = np.array(
            [x if x is not None else [0.0] * dim for x in fastrp_list], dtype=np.float32
        )
        fastrp_tensor: torch.Tensor = torch.tensor(fastrp_arr, dtype=torch.float)

        return torch.cat([scaled_continuous, wcc_tensor, fastrp_tensor], dim=1)

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
        financial_cols: list[str] = [
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
