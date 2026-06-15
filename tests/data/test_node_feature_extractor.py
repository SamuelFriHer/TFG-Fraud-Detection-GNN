"""Unit tests for the NodeFeatureExtractor class."""

import numpy as np
import polars as pl
import pytest
import torch

from src.data.node_feature_extractor import NodeFeatureExtractor


class TestNodeFeatureExtractor:
    """Test suite for NodeFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> NodeFeatureExtractor:
        """Provides a fresh NodeFeatureExtractor instance."""
        return NodeFeatureExtractor()

    @pytest.fixture
    def sample_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Provides sample accounts and transactions DataFrames."""
        accounts_df: pl.DataFrame = pl.DataFrame(
            {
                "Account_ID": ["acc_1", "acc_2", "acc_3"],
            }
        )
        trans_df: pl.DataFrame = pl.DataFrame(
            {
                "From_Acc": ["acc_1", "acc_2", "acc_1"],
                "To_Acc": ["acc_2", "acc_3", "acc_3"],
                "Amount Paid": [10.0, 20.0, 30.0],
                "Amount Received": [10.0, 20.0, 30.0],
            }
        )
        return accounts_df, trans_df

    @pytest.fixture
    def sample_neo4j_df(self) -> pl.DataFrame:
        """Provides a sample Neo4j features DataFrame."""
        return pl.DataFrame(
            {
                "Account_ID": ["acc_1", "acc_2", "acc_3"],
                "wcc_id": [100, 200, 100],
                "pagerank": [0.15, 0.45, 0.30],
                "fastrp_emb": [[0.1] * 64, [0.2] * 64, [0.3] * 64],
            }
        )

    def test_compute_features_without_neo4j(
        self,
        extractor: NodeFeatureExtractor,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Verifies feature extraction and scaling when neo4j_df is None."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data
        features: torch.Tensor = extractor.compute_features(
            accounts_df=accounts_df,
            trans_df=trans_df,
            train_cutoff=3,
            neo4j_df=None,
        )

        assert features.shape == (3, 10)

        for col_idx in range(10):
            col_vals: torch.Tensor = features[:, col_idx]
            mean_val: float = float(col_vals.mean().item())
            std_val: float = float(col_vals.std(correction=0).item())
            assert abs(mean_val) < 1e-5
            if std_val > 1e-5:
                assert abs(std_val - 1.0) < 1e-5

    def test_compute_features_with_neo4j(
        self,
        extractor: NodeFeatureExtractor,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
        sample_neo4j_df: pl.DataFrame,
    ) -> None:
        """Verifies feature scaling applies only to continuous features and PageRank."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data
        features: torch.Tensor = extractor.compute_features(
            accounts_df=accounts_df,
            trans_df=trans_df,
            train_cutoff=3,
            neo4j_df=sample_neo4j_df,
        )

        assert features.shape == (3, 75)

        for col_idx in range(11):
            col_vals: torch.Tensor = features[:, col_idx]
            mean_val: float = float(col_vals.mean().item())
            std_val: float = float(col_vals.std(correction=0).item())
            assert abs(mean_val) < 1e-5
            if std_val > 1e-5:
                assert abs(std_val - 1.0) < 1e-5

        fastrp_vals: torch.Tensor = features[:, 11:]
        assert torch.allclose(fastrp_vals[0], torch.tensor([0.1] * 64))
        assert torch.allclose(fastrp_vals[1], torch.tensor([0.2] * 64))
        assert torch.allclose(fastrp_vals[2], torch.tensor([0.3] * 64))

    def test_apply_log1p(self, extractor: NodeFeatureExtractor) -> None:
        """Verifies that log1p transformation is correctly applied to financial columns."""
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
        data: dict[str, list[float]] = {col: [0.0, 1.0, 10.0, 100.0] for col in financial_cols}
        data["other_col"] = [1.0, 2.0, 3.0, 4.0]

        dummy_df: pl.DataFrame = pl.DataFrame(data)
        result_df: pl.DataFrame = extractor._apply_log1p(dummy_df)

        for col in financial_cols:
            expected: np.ndarray = np.log1p(np.array([0.0, 1.0, 10.0, 100.0]))
            actual: np.ndarray = result_df[col].to_numpy()
            assert np.allclose(actual, expected)

        expected_other: np.ndarray = np.array([1.0, 2.0, 3.0, 4.0])
        actual_other: np.ndarray = result_df["other_col"].to_numpy()
        assert np.allclose(actual_other, expected_other)
