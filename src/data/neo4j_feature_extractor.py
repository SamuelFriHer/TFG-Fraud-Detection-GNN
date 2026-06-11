"""Module for extracting topological features using Neo4j Graph Data Science."""

import os
from typing import Any

import polars as pl
from dotenv import load_dotenv
from graphdatascience import GraphDataScience
from neo4j import Driver, GraphDatabase

from src.utils.logger import ProjectLogger

load_dotenv()


class Neo4jFeatureExtractor:
    """Executes GDS algorithms and extracts topological features to Polars."""

    def __init__(self) -> None:
        """Initializes the GDS client and standard driver."""
        self.uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user: str = os.getenv("NEO4J_USER", "neo4j")
        self.password: str = os.getenv("NEO4J_PASSWORD", "tfg_password")

        self.gds: GraphDataScience = GraphDataScience(self.uri, auth=(self.user, self.password))
        self.driver: Driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.logger: Any = ProjectLogger.get_logger("Neo4jFeatureExtractor")
        self.graph_name: str = "aml_graph"

    def close(self) -> None:
        """Closes connections."""
        self.gds.close()
        self.driver.close()

    def _cleanup_existing_graph(self) -> None:
        """Removes the projected graph if it already exists."""
        exists: bool = bool(self.gds.graph.exists(self.graph_name).exists)
        if exists:
            self.logger.info(f"Graph '{self.graph_name}' exists. Dropping it.")
            g: Any = self.gds.graph.get(self.graph_name)
            g.drop()

    def project_graph(self) -> None:
        """Projects the Neo4j graph into GDS memory."""
        self._cleanup_existing_graph()
        self.logger.info(f"Projecting graph '{self.graph_name}' into GDS...")
        self.gds.graph.project(self.graph_name, "Account", "TRANSACTED")

    def run_algorithms(self, embedding_dim: int = 64) -> None:
        """Executes WCC, PageRank, and FastRP algorithms."""
        self.logger.info("Running WCC (Weakly Connected Components)...")
        g: Any = self.gds.graph.get(self.graph_name)
        self.gds.wcc.mutate(g, mutateProperty="wcc_id")

        self.logger.info("Running PageRank...")
        self.gds.pageRank.mutate(g, mutateProperty="pagerank")

        self.logger.info(f"Running FastRP (dim={embedding_dim})...")
        self.gds.fastRP.mutate(g, mutateProperty="fastrp_emb", embeddingDimension=embedding_dim)

        self.logger.info("Writing properties back to the Neo4j database...")
        self.gds.graph.nodeProperties.write(g, ["wcc_id", "pagerank", "fastrp_emb"])

    def extract_features(self) -> pl.DataFrame:
        """Extracts the computed properties from Neo4j into a Polars DataFrame."""
        self.logger.info("Extracting features from Neo4j into Polars DataFrame...")
        query: str = """
        MATCH (n:Account)
        RETURN n.id AS Account_ID,
               n.wcc_id AS wcc_id,
               n.pagerank AS pagerank,
               n.fastrp_emb AS fastrp_emb
        """
        with self.driver.session() as session:
            result: Any = session.run(query)
            records: list[dict[str, Any]] = result.data()

        df: pl.DataFrame = pl.DataFrame(records)
        return df

    def run_pipeline(self) -> pl.DataFrame:
        """Runs the entire feature extraction pipeline and returns the DataFrame."""
        try:
            self.project_graph()
            self.run_algorithms()
            df: pl.DataFrame = self.extract_features()
        finally:
            self.close()
        self.logger.info("Feature extraction pipeline completed.")
        return df
