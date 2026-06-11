"""Unit tests for the Neo4jFeatureExtractor class."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from src.data.neo4j_feature_extractor import Neo4jFeatureExtractor


class TestNeo4jFeatureExtractor:
    """Test suite for Neo4jFeatureExtractor."""

    @pytest.fixture
    def mock_dependencies(self) -> Generator[tuple[MagicMock, MagicMock], None, None]:
        """Provides mocked GraphDataScience and GraphDatabase driver."""
        with (
            patch("src.data.neo4j_feature_extractor.GraphDataScience") as mock_gds_class,
            patch("src.data.neo4j_feature_extractor.GraphDatabase.driver") as mock_driver_class,
        ):
            mock_gds: MagicMock = MagicMock()
            mock_driver: MagicMock = MagicMock()
            mock_gds_class.return_value = mock_gds
            mock_driver_class.return_value = mock_driver
            yield mock_gds, mock_driver

    @pytest.fixture
    def extractor(self, mock_dependencies: tuple[MagicMock, MagicMock]) -> Neo4jFeatureExtractor:
        """Provides a Neo4jFeatureExtractor instance with mocked dependencies."""
        extractor_instance: Neo4jFeatureExtractor = Neo4jFeatureExtractor()
        return extractor_instance

    def test_extract_features(
        self,
        extractor: Neo4jFeatureExtractor,
        mock_dependencies: tuple[MagicMock, MagicMock],
    ) -> None:
        """Verifies that extract_features extracts data using result.data()."""
        _, mock_driver = mock_dependencies
        session: MagicMock = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = session

        mock_result: MagicMock = MagicMock()
        session.run.return_value = mock_result

        mock_data: list[dict[str, Any]] = [
            {"Account_ID": "acc_1", "wcc_id": 1, "pagerank": 0.15, "fastrp_emb": [0.1, 0.2]},
            {"Account_ID": "acc_2", "wcc_id": 2, "pagerank": 0.35, "fastrp_emb": [0.3, 0.4]},
        ]
        mock_result.data.return_value = mock_data

        df: pl.DataFrame = extractor.extract_features()

        assert session.run.call_count == 1
        assert mock_result.data.call_count == 1
        assert df.shape == (2, 4)
        assert df["Account_ID"].to_list() == ["acc_1", "acc_2"]
        assert df["wcc_id"].to_list() == [1, 2]
        assert df["pagerank"].to_list() == [0.15, 0.35]
        assert df["fastrp_emb"].to_list() == [[0.1, 0.2], [0.3, 0.4]]

    def test_project_graph(
        self,
        extractor: Neo4jFeatureExtractor,
        mock_dependencies: tuple[MagicMock, MagicMock],
    ) -> None:
        """Verifies that project_graph cleans up existing graphs and creates a new projection."""
        mock_gds, _ = mock_dependencies

        # Setup mock for exist check
        mock_gds.graph.exists.return_value.exists = True
        mock_graph: MagicMock = MagicMock()
        mock_gds.graph.get.return_value = mock_graph

        extractor.project_graph()

        mock_gds.graph.exists.assert_called_once_with(extractor.graph_name)
        mock_graph.drop.assert_called_once()
        mock_gds.graph.project.assert_called_once_with(
            extractor.graph_name, "Account", "TRANSACTED"
        )

    def test_run_algorithms(
        self,
        extractor: Neo4jFeatureExtractor,
        mock_dependencies: tuple[MagicMock, MagicMock],
    ) -> None:
        """Verifies that run_algorithms executes mutation and write operations correctly."""
        mock_gds, _ = mock_dependencies
        mock_graph: MagicMock = MagicMock()
        mock_gds.graph.get.return_value = mock_graph

        extractor.run_algorithms(embedding_dim=32)

        mock_gds.graph.get.assert_called_once_with(extractor.graph_name)
        mock_gds.wcc.mutate.assert_called_once_with(mock_graph, mutateProperty="wcc_id")
        mock_gds.pageRank.mutate.assert_called_once_with(mock_graph, mutateProperty="pagerank")
        mock_gds.fastRP.mutate.assert_called_once_with(
            mock_graph, mutateProperty="fastrp_emb", embeddingDimension=32
        )
        mock_gds.graph.nodeProperties.write.assert_called_once_with(
            mock_graph, ["wcc_id", "pagerank", "fastrp_emb"]
        )
