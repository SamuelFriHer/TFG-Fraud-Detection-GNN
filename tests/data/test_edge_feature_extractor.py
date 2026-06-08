"""Unit tests for the EdgeFeatureExtractor class."""

import polars as pl
import pytest
import torch

from src.data.edge_feature_extractor import EdgeFeatureExtractor


class TestEdgeFeatureExtractor:
    """Test suite for EdgeFeatureExtractor."""

    @pytest.fixture
    def extractor(self) -> EdgeFeatureExtractor:
        """Provides a fresh EdgeFeatureExtractor instance."""
        return EdgeFeatureExtractor()

    @pytest.fixture
    def sample_trans_df(self) -> pl.DataFrame:
        """Provides a sample transactions DataFrame."""
        return pl.DataFrame(
            {
                "From Bank": [1, 2, 1],
                "Account": ["A", "B", "A"],
                "To Bank": [2, 3, 2],
                "Account.1": ["B", "C", "B"],
                "Amount Received": [100.0, 200.0, 150.0],
                "Amount Paid": [98.0, 195.0, 147.0],
                "Receiving Currency": ["USD", "EUR", "USD"],
                "Payment Currency": ["USD", "EUR", "USD"],
                "Payment Format": ["Wire", "ACH", "Wire"],
                "Is Laundering": [0, 0, 0],
                "Timestamp": [1.0, 2.0, 3.0],
                "From_Acc": ["1_A", "2_B", "1_A"],
                "To_Acc": ["2_B", "3_C", "2_B"],
            }
        )

    def test_extract_features_shape_and_content(
        self, extractor: EdgeFeatureExtractor, sample_trans_df: pl.DataFrame
    ) -> None:
        """Verifies that features are extracted with correct dimensions and scaling."""
        features: torch.Tensor = extractor.extract_features(sample_trans_df)

        # Remaining columns (6): Amount Received, Amount Paid, Receiving Currency,
        # Payment Currency, Payment Format, Timestamp.
        assert features.shape == (3, 6)

        # Verify that categorical columns are registered in self.edge_encoders
        assert "Receiving Currency" in extractor.edge_encoders
        assert extractor.edge_encoders["Receiving Currency"] == ["EUR", "USD"]

        # Verify normalization (mean should be close to 0, std close to 1)
        for col_idx in range(6):
            col_vals: torch.Tensor = features[:, col_idx]
            mean_val: float = float(col_vals.mean().item())
            std_val: float = float(col_vals.std(correction=0).item())
            assert abs(mean_val) < 1e-5
            if std_val > 1e-5:
                assert abs(std_val - 1.0) < 1e-5

    def test_encode_dataframe_unhandled_string_column(
        self, extractor: EdgeFeatureExtractor
    ) -> None:
        """Verifies that ValueError is raised for unhandled string columns."""
        df: pl.DataFrame = pl.DataFrame(
            {
                "UnhandledCol": ["A", "B", "C"],
            }
        )
        with pytest.raises(ValueError, match="Unhandled string column: UnhandledCol"):
            extractor._encode_dataframe(df, cols=[])
