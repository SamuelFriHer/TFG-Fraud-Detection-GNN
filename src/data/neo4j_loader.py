"""Module for loading accounts and transactions data into Neo4j."""

import os

import polars as pl
from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.utils.logger import ProjectLogger

load_dotenv()


class Neo4jLoader:
    """Handles data ingestion from Polars DataFrames to Neo4j."""

    def __init__(self) -> None:
        """Initializes the Neo4j driver using environment variables."""
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "tfg_password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.logger = ProjectLogger.get_logger("Neo4jLoader")

        max_retries: int = 2 if "PYTEST_CURRENT_TEST" in os.environ else 30
        self._wait_for_connection(max_retries=max_retries)

    def _wait_for_connection(self, max_retries: int = 30, delay: float = 2.0) -> None:
        """Waits for the Neo4j server to be available and accepting connections."""
        import time

        from neo4j.exceptions import ServiceUnavailable

        for attempt in range(max_retries):
            try:
                self.driver.verify_connectivity()
                self.logger.info("Successfully connected to Neo4j database.")
                return
            except ServiceUnavailable as e:
                if attempt == max_retries - 1:
                    self.logger.critical("Could not connect to Neo4j after multiple retries.")
                    raise e
                self.logger.warning(
                    f"Neo4j not ready (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        self.driver.close()

    def clean_db(self) -> None:
        """Removes all nodes and relationships from the database in batches to avoid OOM."""
        self.logger.info("Cleaning Neo4j database...")
        with self.driver.session() as session:
            try:
                session.run(
                    "CALL apoc.periodic.iterate("
                    "'MATCH ()-[r:TRANSACTED]->() RETURN r', "
                    "'DELETE r', "
                    "{batchSize: 50000, parallel: false}"
                    ")"
                )
                session.run(
                    "CALL apoc.periodic.iterate("
                    "'MATCH (n:Account) RETURN n', "
                    "'DELETE n', "
                    "{batchSize: 50000, parallel: false}"
                    ")"
                )
            except Exception as e:
                self.logger.warning(f"Batched cleanup failed: {e}. Falling back to detach delete.")
                session.run("MATCH (n) DETACH DELETE n")

            try:
                session.run("DROP CONSTRAINT account_id_unique IF EXISTS")
            except Exception as e:
                self.logger.warning(f"Could not drop constraint: {e}")

    def create_constraints(self) -> None:
        """Creates unique constraints for Account nodes."""
        self.logger.info("Creating constraints...")
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT account_id_unique IF NOT EXISTS "
                "FOR (a:Account) REQUIRE a.id IS UNIQUE"
            )

    def load_accounts(self, accounts_df: pl.DataFrame, batch_size: int = 10000) -> None:
        """Loads accounts into Neo4j in batches."""
        self.logger.info(f"Loading {len(accounts_df)} accounts into Neo4j...")

        query = """
        UNWIND $batch AS acc
        MERGE (a:Account {id: acc.id})
        """

        with self.driver.session() as session:
            for i in range(0, len(accounts_df), batch_size):
                batch_df: pl.DataFrame = accounts_df.slice(i, batch_size)
                batch: list[dict[str, str]] = batch_df.select(
                    pl.col("Account_ID").alias("id")
                ).to_dicts()
                session.run(query, batch=batch)

    def load_transactions(self, trans_df: pl.DataFrame, batch_size: int = 10000) -> None:
        """Loads transactions as relationships into Neo4j in batches."""
        self.logger.info(f"Loading {len(trans_df)} transactions into Neo4j...")

        query = """
        UNWIND $batch AS tx
        MATCH (src:Account {id: tx.from_acc})
        MATCH (dst:Account {id: tx.to_acc})
        CREATE (src)-[:TRANSACTED {amount: tx.amount, currency: tx.currency}]->(dst)
        """

        with self.driver.session() as session:
            for i in range(0, len(trans_df), batch_size):
                batch_df: pl.DataFrame = trans_df.slice(i, batch_size)
                batch: list[dict[str, str | float]] = batch_df.select(
                    [
                        pl.col("From_Acc").alias("from_acc"),
                        pl.col("To_Acc").alias("to_acc"),
                        pl.col("Amount Paid").cast(pl.Float64).alias("amount"),
                        pl.col("Payment Currency").alias("currency"),
                    ]
                ).to_dicts()
                session.run(query, batch=batch)

    def run_pipeline(self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame) -> None:
        """Executes the complete data ingestion pipeline."""
        try:
            self.clean_db()
            self.create_constraints()
            self.load_accounts(accounts_df)
            self.load_transactions(trans_df)
        finally:
            self.close()
        self.logger.info("Data ingestion to Neo4j completed.")
