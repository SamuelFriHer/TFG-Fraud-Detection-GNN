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

    def test_extract_features_unsupported_string_column(
        self, extractor: EdgeFeatureExtractor, sample_trans_df: pl.DataFrame
    ) -> None:
        """Verifies that extract_features raises ValueError for an unsupported string column."""
        unsupported_df: pl.DataFrame = sample_trans_df.with_columns(
            pl.Series("UnsupportedCol", ["X", "Y", "Z"])
        )
        with pytest.raises(ValueError, match="Unhandled string column: UnsupportedCol"):
            extractor.extract_features(unsupported_df)

    def test_scale_tensor_dummy_tensor(self, extractor: EdgeFeatureExtractor) -> None:
        """Verifies that _scale_tensor correctly scales a dummy tensor to mean ~0 and std ~1.

        Also checks that constant columns with zero standard deviation are correctly handled
        without generating NaN values.
        """
        dummy_tensor: torch.Tensor = torch.tensor(
            [
                [1.0, 10.0, 5.0],
                [2.0, 20.0, 5.0],
                [3.0, 30.0, 5.0],
                [4.0, 40.0, 5.0],
            ],
            dtype=torch.float32,
        )
        scaled_tensor: torch.Tensor = extractor._scale_tensor(dummy_tensor)

        mean_vals: torch.Tensor = scaled_tensor.mean(dim=0)
        std_vals: torch.Tensor = scaled_tensor.std(dim=0, correction=0)

        for idx in [0, 1]:
            assert abs(float(mean_vals[idx].item())) < 1e-5
            assert abs(float(std_vals[idx].item()) - 1.0) < 1e-5

        assert abs(float(mean_vals[2].item())) < 1e-5
        constant_col_std: float = float(scaled_tensor[:, 2].std(correction=0).item())
        assert abs(constant_col_std) < 1e-5
        assert torch.all(scaled_tensor[:, 2] == 0.0)
