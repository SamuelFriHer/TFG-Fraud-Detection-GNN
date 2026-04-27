"""Unit tests for the DataPreprocessor class using the ZOMBIES pattern."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    """Test suite for DataPreprocessor following ZOMBIES pattern."""

    @pytest.fixture
    def preprocessor(self) -> DataPreprocessor:
        """Provides a fresh DataPreprocessor instance."""
        return DataPreprocessor()

    @pytest.fixture
    def sample_df(self) -> pl.DataFrame:
        """Provides a standard sample DataFrame for testing."""
        return pl.DataFrame(
            {
                "id": [1, 2, 3],
                "feature": ["A", "B", "A"],
                "value": [10.0, 20.0, None],
                "target": [0, 1, 0],
            }
        )

    # --- ZERO ---

    def test_clean_data_empty(self, preprocessor: DataPreprocessor) -> None:
        """Zero: Verifies cleaning an empty DataFrame."""
        empty_df = pl.DataFrame()
        result = preprocessor.clean_data(empty_df)
        assert result.height == 0
        assert result.width == 0

    def test_encode_features_no_categorical(
        self, preprocessor: DataPreprocessor, sample_df: pl.DataFrame
    ) -> None:
        """Zero: Verifies encoding when no categorical columns are specified."""
        result = preprocessor.encode_features(sample_df, [])
        assert result.equals(sample_df)

    # --- ONE ---

    def test_clean_data_single_row_no_nulls(self, preprocessor: DataPreprocessor) -> None:
        """One: Verifies cleaning a single row without nulls."""
        df = pl.DataFrame({"a": [1]})
        result = preprocessor.clean_data(df)
        assert result.height == 1

    def test_clean_data_single_row_with_nulls(self, preprocessor: DataPreprocessor) -> None:
        """One: Verifies cleaning a single row with nulls."""
        df = pl.DataFrame({"a": [None]})
        result = preprocessor.clean_data(df)
        assert result.height == 0

    def test_encode_features_single_column(
        self, preprocessor: DataPreprocessor, sample_df: pl.DataFrame
    ) -> None:
        """One: Verifies encoding a single categorical column."""
        result = preprocessor.encode_features(sample_df, ["feature"])
        assert result["feature"].dtype == pl.Int64
        assert len(preprocessor.encoders) == 1
        assert "feature" in preprocessor.encoders

    # --- MANY ---

    def test_encode_features_multiple_columns(
        self, preprocessor: DataPreprocessor, sample_df: pl.DataFrame
    ) -> None:
        """Many: Verifies encoding multiple categorical columns."""
        # Add another categorical column
        df = sample_df.with_columns(pl.Series("type", ["X", "Y", "X"]))
        result = preprocessor.encode_features(df, ["feature", "type"])
        assert result["feature"].dtype == pl.Int64
        assert result["type"].dtype == pl.Int64
        assert len(preprocessor.encoders) == 2

    def test_split_data_standard(self, preprocessor: DataPreprocessor) -> None:
        """Many: Verifies splitting a multi-row dataset."""
        # Create a df with enough rows for a 60/20/20 split (min 5 rows for 20% to be 1)
        df = pl.DataFrame(
            {"feat1": np.random.rand(10), "feat2": np.random.rand(10), "target": [0, 1] * 5}
        )
        x_train, x_val, x_test, y_train, y_val, y_test = preprocessor.split_data(df, "target")

        assert len(x_train) == 6
        assert len(x_val) == 2
        assert len(x_test) == 2
        assert len(y_train) == 6
        assert isinstance(x_train, np.ndarray)

    # --- BOUNDARY ---

    def test_clean_data_all_nulls(self, preprocessor: DataPreprocessor) -> None:
        """Boundary: Verifies behavior when all rows contain nulls."""
        df = pl.DataFrame({"a": [None, None], "b": [1, None]})
        result = preprocessor.clean_data(df)
        assert result.height == 0

    def test_split_data_small_sample(self, preprocessor: DataPreprocessor) -> None:
        """Boundary: Verifies splitting with minimum possible rows."""
        df = pl.DataFrame({"f": [1, 2, 3, 4, 5], "t": [0, 1, 0, 1, 0]})
        # 60% of 5 is 3. 40% left is 2. 50% of 2 is 1. So 3/1/1 split.
        x_train, x_val, x_test, _, _, _ = preprocessor.split_data(df, "t")
        assert len(x_train) == 3
        assert len(x_val) == 1
        assert len(x_test) == 1

    # --- INTERFACE ---

    def test_load_data_direct_path(self, preprocessor: DataPreprocessor, tmp_path: Path) -> None:
        """Interface: Verifies loading a CSV by direct file path."""
        csv_file = tmp_path / "data.csv"
        pl.DataFrame({"a": [1]}).write_csv(csv_file)
        df = preprocessor.load_data(str(csv_file))
        assert df.height == 1
        assert "a" in df.columns

    def test_load_data_from_dir(self, preprocessor: DataPreprocessor, tmp_path: Path) -> None:
        """Interface: Verifies loading by providing a directory path."""
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        csv_file = dataset_dir / "HI-Small_Trans.csv"
        pl.DataFrame({"b": [2]}).write_csv(csv_file)

        df = preprocessor.load_data(str(dataset_dir), dataset_prefix="HI-Small")
        assert df.height == 1
        assert "b" in df.columns

    # --- EXCEPTIONS ---

    def test_load_data_path_not_exists(self, preprocessor: DataPreprocessor) -> None:
        """Exceptions: Verifies FileNotFoundError for invalid paths."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data("/non/existent/path")

    def test_load_data_missing_csv_in_dir(
        self, preprocessor: DataPreprocessor, tmp_path: Path
    ) -> None:
        """Exceptions: Verifies FileNotFoundError when prefix file is missing."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            preprocessor.load_data(str(empty_dir), dataset_prefix="Missing")
