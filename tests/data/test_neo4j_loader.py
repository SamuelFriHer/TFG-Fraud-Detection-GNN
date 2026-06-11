"""Unit tests for the Neo4jLoader class."""

from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data.neo4j_loader import Neo4jLoader


class TestNeo4jLoader:
    """Test suite for Neo4jLoader."""

    @pytest.fixture
    def mock_driver(self) -> MagicMock:
        """Provides a mocked Neo4j driver."""
        driver: MagicMock = MagicMock()
        driver.verify_connectivity = MagicMock()
        return driver

    @pytest.fixture
    def loader(self, mock_driver: MagicMock) -> Neo4jLoader:
        """Provides a Neo4jLoader instance with a mocked driver."""
        with patch("src.data.neo4j_loader.GraphDatabase.driver", return_value=mock_driver):
            loader_instance: Neo4jLoader = Neo4jLoader()
            return loader_instance

    def test_load_accounts(self, loader: Neo4jLoader, mock_driver: MagicMock) -> None:
        """Verifies that load_accounts formats and loads account data in batches."""
        accounts_df: pl.DataFrame = pl.DataFrame({"Account_ID": ["acc_1", "acc_2", "acc_3"]})

        session: MagicMock = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = session

        loader.load_accounts(accounts_df, batch_size=2)

        assert session.run.call_count == 2

        first_call_args = session.run.call_args_list[0]
        assert "UNWIND $batch AS acc" in first_call_args[0][0]
        assert first_call_args[1]["batch"] == [{"id": "acc_1"}, {"id": "acc_2"}]

        second_call_args = session.run.call_args_list[1]
        assert second_call_args[1]["batch"] == [{"id": "acc_3"}]

    def test_load_transactions(self, loader: Neo4jLoader, mock_driver: MagicMock) -> None:
        """Verifies that load_transactions formats and loads transaction data in batches."""
        trans_df: pl.DataFrame = pl.DataFrame(
            {
                "From_Acc": ["acc_1", "acc_2"],
                "To_Acc": ["acc_2", "acc_3"],
                "Amount Paid": [10.5, 20.0],
                "Payment Currency": ["USD", "EUR"],
            }
        )

        session: MagicMock = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = session

        loader.load_transactions(trans_df, batch_size=1)

        assert session.run.call_count == 2

        first_call_args = session.run.call_args_list[0]
        assert "UNWIND $batch AS tx" in first_call_args[0][0]
        assert first_call_args[1]["batch"] == [
            {"from_acc": "acc_1", "to_acc": "acc_2", "amount": 10.5, "currency": "USD"}
        ]

        second_call_args = session.run.call_args_list[1]
        assert second_call_args[1]["batch"] == [
            {"from_acc": "acc_2", "to_acc": "acc_3", "amount": 20.0, "currency": "EUR"}
        ]
