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

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        self.driver.close()

    def clean_db(self) -> None:
        """Removes all nodes and relationships from the database."""
        self.logger.info("Cleaning Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            # Drop existing constraints just in case
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
                batch_df = accounts_df.slice(i, batch_size)
                batch = [{"id": row["Account_ID"]} for row in batch_df.to_dicts()]
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
                batch_df = trans_df.slice(i, batch_size)
                batch = [
                    {
                        "from_acc": row["From_Acc"],
                        "to_acc": row["To_Acc"],
                        "amount": float(row["Amount Paid"]),
                        "currency": row["Payment Currency"],
                    }
                    for row in batch_df.to_dicts()
                ]
                session.run(query, batch=batch)

    def run_pipeline(self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame) -> None:
        """Executes the complete data ingestion pipeline."""
        self.clean_db()
        self.create_constraints()
        self.load_accounts(accounts_df)
        self.load_transactions(trans_df)
        self.close()
        self.logger.info("Data ingestion to Neo4j completed.")
