"""Unit tests for the AMLGraphBuilder class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from src.data.graph_builder import AMLGraphBuilder


class TestAMLGraphBuilder:
    """Test suite for AMLGraphBuilder."""

    @pytest.fixture
    def builder(self) -> AMLGraphBuilder:
        """Provides a fresh AMLGraphBuilder instance."""
        return AMLGraphBuilder()

    @pytest.fixture
    def sample_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Provides sample accounts and transactions DataFrames."""
        accounts_df: pl.DataFrame = pl.DataFrame(
            {
                "Bank ID": [1, 2],
                "Account Number": ["acc_1", "acc_2"],
            }
        )
        trans_df: pl.DataFrame = pl.DataFrame(
            {
                "From Bank": [1, 2],
                "Account": ["acc_1", "acc_2"],
                "To Bank": [2, 1],
                "Account.1": ["acc_2", "acc_1"],
                "Timestamp": ["2026/06/15 12:00", "2026/06/15 12:05"],
                "Amount Paid": [10.0, 20.0],
                "Payment Currency": ["USD", "USD"],
                "Is Laundering": [0, 1],
            }
        )
        return accounts_df, trans_df

    def test_find_csv_success(self, builder: AMLGraphBuilder, tmp_path: Path) -> None:
        """Verifies that _find_csv returns the correct path when file exists."""
        prefix: str = "test"
        suffix: str = "accounts"
        file_path: Path = tmp_path / f"{prefix}_{suffix}.csv"
        file_path.touch()

        found_path: Path = builder._find_csv(str(tmp_path), prefix, suffix)
        assert found_path == file_path

    def test_find_csv_not_found(self, builder: AMLGraphBuilder, tmp_path: Path) -> None:
        """Verifies that _find_csv raises FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            builder._find_csv(str(tmp_path), "non_existent", "accounts")

    def test_load_data(self, builder: AMLGraphBuilder, tmp_path: Path) -> None:
        """Verifies that _load_data loads both CSVs and renames duplicated columns."""
        prefix: str = "test"
        accounts_path: Path = tmp_path / f"{prefix}_accounts.csv"
        trans_path: Path = tmp_path / f"{prefix}_Trans.csv"

        accounts_df: pl.DataFrame = pl.DataFrame({"Bank ID": [1], "Account Number": ["acc_1"]})
        trans_df: pl.DataFrame = pl.DataFrame({"From Bank": [1], "Account_duplicated_0": ["acc_2"]})

        accounts_df.write_csv(accounts_path)
        trans_df.write_csv(trans_path)

        loaded_acc, loaded_trans = builder._load_data(str(tmp_path), prefix)
        assert "Account.1" in loaded_trans.columns
        assert "Account_duplicated_0" not in loaded_trans.columns
        assert len(loaded_acc) == 1
        assert len(loaded_trans) == 1

    def test_prepare_accounts_and_transactions(
        self,
        builder: AMLGraphBuilder,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Verifies that identifiers are correctly created and transactions prepared."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data

        p_acc, p_trans = builder._prepare_accounts_and_transactions(accounts_df, trans_df)

        assert "Account_ID" in p_acc.columns
        assert "From_Acc" in p_trans.columns
        assert "To_Acc" in p_trans.columns
        assert set(p_acc["Account_ID"].to_list()) == {"1_acc_1", "2_acc_2"}
        assert builder.account_id_map["1_acc_1"] in (0, 1)
        assert builder.account_id_map["2_acc_2"] in (0, 1)

    def test_compute_edge_index_and_masks(
        self,
        builder: AMLGraphBuilder,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Verifies correct calculation of edge indices and split masks."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data
        _, p_trans = builder._prepare_accounts_and_transactions(accounts_df, trans_df)

        edge_index, train_mask, val_mask, test_mask = builder._compute_edge_index_and_masks(
            p_trans, test_size=0.5
        )

        assert edge_index.shape == (2, 2)
        # Verify edge mappings are indices
        assert edge_index[0, 0].item() == builder.account_id_map["1_acc_1"]
        assert edge_index[1, 0].item() == builder.account_id_map["2_acc_2"]
        assert train_mask.shape == (2,)
        assert val_mask.shape == (2,)
        assert test_mask.shape == (2,)

    def test_create_split_masks(self, builder: AMLGraphBuilder) -> None:
        """Verifies the indices of split masks."""
        train_mask, val_mask, test_mask = builder._create_split_masks(n_edges=4, test_size=0.5)
        # test_size = 0.5 -> train_cutoff = 4 * 0.5 = 2. val_cutoff = 4 * 0.75 = 3
        # train: [0, 1], val: [2], test: [3]
        assert train_mask.tolist() == [True, True, False, False]
        assert val_mask.tolist() == [False, False, True, False]
        assert test_mask.tolist() == [False, False, False, True]

    def test_build_graph_success(
        self,
        builder: AMLGraphBuilder,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Verifies full build_graph flow when Neo4j runs successfully."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data

        mock_neo4j_df: pl.DataFrame = pl.DataFrame({"Account_ID": ["1_acc_1", "2_acc_2"]})

        with (
            patch.object(builder, "_load_data", return_value=(accounts_df, trans_df)),
            patch("src.data.graph_builder.Neo4jLoader") as mock_loader_cls,
            patch("src.data.graph_builder.Neo4jFeatureExtractor") as mock_extractor_cls,
        ):
            mock_loader = mock_loader_cls.return_value
            mock_extractor = mock_extractor_cls.return_value
            mock_extractor.run_pipeline.return_value = mock_neo4j_df

            # Mock extractors inside builder
            builder.node_extractor = MagicMock()
            builder.node_extractor.compute_features.return_value = torch.ones((2, 5))
            builder.edge_extractor = MagicMock()
            builder.edge_extractor.extract_features.return_value = torch.zeros((2, 3))

            data: Data = builder.build_graph(dataset_dir="dummy", prefix="dummy", test_size=0.5)

            mock_loader.run_pipeline.assert_called_once()
            mock_extractor.run_pipeline.assert_called_once()
            builder.node_extractor.compute_features.assert_called_once()
            builder.edge_extractor.extract_features.assert_called_once()

            assert isinstance(data, Data)
            assert data.x.shape == (2, 5)
            assert data.edge_attr.shape == (2, 3)
            assert data.edge_index.shape == (2, 2)
            assert torch.equal(data.y, torch.tensor([0, 1], dtype=torch.long))

    def test_build_graph_neo4j_failure_fallback(
        self,
        builder: AMLGraphBuilder,
        sample_data: tuple[pl.DataFrame, pl.DataFrame],
    ) -> None:
        """Verifies build_graph falls back to basic features if Neo4j raises an exception."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = sample_data

        with (
            patch.object(builder, "_load_data", return_value=(accounts_df, trans_df)),
            patch(
                "src.data.graph_builder.Neo4jLoader",
                side_effect=GraphBuilderTestError("Neo4j is down"),
            ),
        ):
            builder.node_extractor = MagicMock()
            builder.node_extractor.compute_features.return_value = torch.ones((2, 5))
            builder.edge_extractor = MagicMock()
            builder.edge_extractor.extract_features.return_value = torch.zeros((2, 3))

            data: Data = builder.build_graph(dataset_dir="dummy", prefix="dummy", test_size=0.5)

            # verify it called compute_features with neo4j_df = None
            args, kwargs = builder.node_extractor.compute_features.call_args
            assert kwargs.get("neo4j_df") is None or args[3] is None

            assert isinstance(data, Data)
            assert data.x.shape == (2, 5)


class GraphBuilderTestError(Exception):
    """Custom exception used for simulating runtime errors during tests."""

    pass
