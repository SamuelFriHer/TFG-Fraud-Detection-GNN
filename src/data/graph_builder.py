"""Module responsible for building graphs from AML tabular data."""

from pathlib import Path

import polars as pl
import torch
from torch_geometric.data import Data

from src.data.edge_feature_extractor import EdgeFeatureExtractor
from src.data.node_feature_extractor import NodeFeatureExtractor


class AMLGraphBuilder:
    """Builds a homogeneous graph from AML accounts and transactions."""

    def __init__(self) -> None:
        """Initializes the graph builder with the necessary mappings."""
        self.node_extractor = NodeFeatureExtractor()
        self.edge_extractor = EdgeFeatureExtractor()
        self.node_encoders: dict[str, list[str]] = {}
        self.account_id_map: dict[str, int] = {}

    @property
    def edge_encoders(self) -> dict[str, list[str]]:
        """Exposes the edge encoders from the edge extractor."""
        return self.edge_extractor.edge_encoders

    def _find_csv(self, directory: str, prefix: str, suffix: str) -> Path:
        """Searches for a CSV file with a specific prefix and suffix."""
        expected_name: str = f"{prefix}_{suffix}.csv"
        path: Path = Path(directory) / expected_name
        if not path.exists():
            raise FileNotFoundError(f"File not found: {expected_name}")
        return path

    def _load_data(self, dataset_dir: str, prefix: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Loads accounts and transactions CSV files."""
        accounts_path: Path = self._find_csv(dataset_dir, prefix, "accounts")
        trans_path: Path = self._find_csv(dataset_dir, prefix, "Trans")

        accounts_df: pl.DataFrame = pl.read_csv(str(accounts_path))
        trans_df: pl.DataFrame = pl.read_csv(str(trans_path))

        if "Account_duplicated_0" in trans_df.columns:
            trans_df = trans_df.rename({"Account_duplicated_0": "Account.1"})

        return accounts_df, trans_df

    def _prepare_accounts_and_transactions(
        self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Creates unique string identifiers and sorts transactions chronologically."""
        unique_accounts: list[str]
        accounts_df, unique_accounts = self._prepare_accounts(accounts_df)
        trans_df_sorted: pl.DataFrame = self._prepare_transactions(trans_df, unique_accounts)
        return accounts_df, trans_df_sorted

    def _prepare_accounts(self, accounts_df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        """Prepares accounts DataFrame and returns unique account list."""
        acc_id: pl.Series = accounts_df["Bank ID"].cast(pl.String) + "_" + accounts_df["Account Number"]
        accounts_df = accounts_df.with_columns(acc_id.alias("Account_ID"))
        unique_accounts: list[str] = acc_id.unique().to_list()
        self.account_id_map = {acc: idx for idx, acc in enumerate(unique_accounts)}
        return accounts_df, unique_accounts

    def _prepare_transactions(
        self, trans_df: pl.DataFrame, unique_accounts: list[str]
    ) -> pl.DataFrame:
        """Prepares and filters transactions DataFrame."""
        src_accounts: pl.Series = trans_df["From Bank"].cast(pl.String) + "_" + trans_df["Account"]
        dst_accounts: pl.Series = trans_df["To Bank"].cast(pl.String) + "_" + trans_df["Account.1"]
        trans_df = trans_df.with_columns(
            [src_accounts.alias("From_Acc"), dst_accounts.alias("To_Acc")]
        )
        valid_mask: pl.Series = trans_df["From_Acc"].is_in(unique_accounts) & trans_df["To_Acc"].is_in(
            unique_accounts
        )
        return (
            trans_df.filter(valid_mask)
            .with_columns(
                pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=False)
            )
            .sort("Timestamp")
        )

    def _compute_edge_index_and_masks(
        self, trans_df: pl.DataFrame, test_size: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps edges to node indices and computes chronological train/val/test masks."""
        src_idx: pl.Series = trans_df["From_Acc"].replace_strict(
            self.account_id_map, default=None, return_dtype=pl.Int64
        )
        dst_idx: pl.Series = trans_df["To_Acc"].replace_strict(
            self.account_id_map, default=None, return_dtype=pl.Int64
        )
        edge_index: torch.Tensor = torch.stack(
            [
                torch.tensor(src_idx.to_numpy(), dtype=torch.long),
                torch.tensor(dst_idx.to_numpy(), dtype=torch.long),
            ],
            dim=0,
        )
        train_mask: torch.Tensor
        val_mask: torch.Tensor
        test_mask: torch.Tensor
        train_mask, val_mask, test_mask = self._create_split_masks(len(trans_df), test_size)
        return edge_index, train_mask, val_mask, test_mask

    def _create_split_masks(
        self, n_edges: int, test_size: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates train, validation, and test boolean masks."""
        train_cutoff: int = int(n_edges * (1.0 - test_size))
        val_cutoff: int = int(n_edges * (1.0 - test_size / 2))

        train_mask: torch.Tensor = torch.zeros(n_edges, dtype=torch.bool)
        val_mask: torch.Tensor = torch.zeros(n_edges, dtype=torch.bool)
        test_mask: torch.Tensor = torch.zeros(n_edges, dtype=torch.bool)

        train_mask[:train_cutoff] = True
        val_mask[train_cutoff:val_cutoff] = True
        test_mask[val_cutoff:] = True

        return train_mask, val_mask, test_mask

    def build_graph(self, dataset_dir: str, prefix: str, test_size: float = 0.4) -> Data:
        """Builds and returns a PyTorch Geometric Data object."""
        accounts_df: pl.DataFrame
        trans_df: pl.DataFrame
        accounts_df, trans_df = self._load_data(dataset_dir, prefix)
        accounts_df, trans_df = self._prepare_accounts_and_transactions(accounts_df, trans_df)
        n_edges: int = len(trans_df)
        train_cutoff: int = int(n_edges * (1.0 - test_size))
        x: torch.Tensor = self.node_extractor.compute_features(accounts_df, trans_df, train_cutoff)
        edge_index: torch.Tensor
        train_mask: torch.Tensor
        val_mask: torch.Tensor
        test_mask: torch.Tensor
        edge_index, train_mask, val_mask, test_mask = self._compute_edge_index_and_masks(
            trans_df, test_size
        )
        edge_attr_scaled: torch.Tensor = self.edge_extractor.extract_features(trans_df)
        y: torch.Tensor = torch.tensor(trans_df["Is Laundering"].to_numpy(), dtype=torch.long)
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_scaled,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
