"""Módulo encargado de la construcción de grafos a partir de los datos tabulares AML."""

from pathlib import Path

import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler  # type: ignore
from torch_geometric.data import Data  # type: ignore


class AMLGraphBuilder:
    """Construye un grafo homogéneo a partir de las transacciones y cuentas AML."""

    def __init__(self) -> None:
        """Inicializa el constructor de grafos con los mapeos necesarios."""
        self.node_encoders: dict[str, LabelEncoder] = {}
        self.edge_encoders: dict[str, LabelEncoder] = {}
        self.account_id_map: dict[str, int] = {}

    def _find_csv(self, directory: str, prefix: str, suffix: str) -> Path:
        """Busca un fichero CSV con un prefijo y sufijo específicos."""
        expected_name = f"{prefix}_{suffix}.csv"
        path = Path(directory) / expected_name
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el fichero: {expected_name}")
        return path

    def _encode_dataframe(
        self, df: pl.DataFrame, cols: list[str], encoders: dict[str, LabelEncoder]
    ) -> torch.Tensor:
        """Codifica un DataFrame a un tensor numérico."""
        encoded_dict: dict[str, pl.Series] = {}
        for col in df.columns:
            if col in cols:
                if col not in encoders:
                    encoders[col] = LabelEncoder()
                pandas_series = df[col].to_pandas()
                encoded_vals = encoders[col].fit_transform(pandas_series)
                encoded_dict[col] = pl.Series(col, encoded_vals)
            else:
                # Tratar como numérico directo
                if df.schema[col] == pl.String:
                    # Parse dates or drop unhandled strings
                    if col == "Timestamp":
                        encoded_dict[col] = df[col].cast(pl.Float32)
                    else:
                        raise ValueError(f"Columna string no manejada: {col}")
                else:
                    encoded_dict[col] = df[col].cast(pl.Float32)

        encoded_df = pl.DataFrame(encoded_dict)
        return torch.tensor(encoded_df.to_numpy(), dtype=torch.float)

    def _prepare_accounts_and_transactions(
        self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Creates unique string identifiers and sorts transactions chronologically."""
        acc_id = accounts_df["Bank ID"].cast(pl.String) + "_" + accounts_df["Account Number"]
        accounts_df = accounts_df.with_columns(acc_id.alias("Account_ID"))

        unique_accounts = acc_id.unique().to_list()
        self.account_id_map = {acc: idx for idx, acc in enumerate(unique_accounts)}

        src_accounts = trans_df["From Bank"].cast(pl.String) + "_" + trans_df["Account"]
        dst_accounts = trans_df["To Bank"].cast(pl.String) + "_" + trans_df["Account.1"]
        trans_df = trans_df.with_columns(
            [src_accounts.alias("From_Acc"), dst_accounts.alias("To_Acc")]
        )

        valid_mask = trans_df["From_Acc"].is_in(unique_accounts) & trans_df["To_Acc"].is_in(
            unique_accounts
        )
        trans_df_valid = trans_df.filter(valid_mask)

        trans_df_sorted = trans_df_valid.with_columns(
            pl.col("Timestamp").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=False)
        ).sort("Timestamp")

        return accounts_df, trans_df_sorted

    def _compute_node_features(
        self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame, train_cutoff: int
    ) -> torch.Tensor:
        """Computes structural and financial statistics for nodes using training data only."""
        train_trans = trans_df.slice(0, train_cutoff)

        out_agg = train_trans.group_by("From_Acc").agg(
            [
                pl.len().alias("out_degree"),
                pl.col("Amount Paid").sum().alias("total_sent"),
                pl.col("Amount Paid").mean().alias("avg_sent"),
                pl.col("Amount Paid").max().alias("max_sent"),
                pl.col("Amount Paid").min().alias("min_sent"),
            ]
        )

        in_agg = train_trans.group_by("To_Acc").agg(
            [
                pl.len().alias("in_degree"),
                pl.col("Amount Received").sum().alias("total_received"),
                pl.col("Amount Received").mean().alias("avg_received"),
                pl.col("Amount Received").max().alias("max_received"),
                pl.col("Amount Received").min().alias("min_received"),
            ]
        )

        node_df = accounts_df.select(["Account_ID"])
        node_df = node_df.join(out_agg, left_on="Account_ID", right_on="From_Acc", how="left")
        node_df = node_df.join(in_agg, left_on="Account_ID", right_on="To_Acc", how="left")
        node_df = node_df.fill_null(0.0)

        financial_cols = [
            "total_sent",
            "avg_sent",
            "max_sent",
            "min_sent",
            "total_received",
            "avg_received",
            "max_received",
            "min_received",
        ]
        node_df = node_df.with_columns([pl.col(c).log1p() for c in financial_cols])

        feature_cols = [
            "out_degree",
            "total_sent",
            "avg_sent",
            "max_sent",
            "min_sent",
            "in_degree",
            "total_received",
            "avg_received",
            "max_received",
            "min_received",
        ]
        x_tensor = torch.tensor(node_df.select(feature_cols).to_numpy(), dtype=torch.float)
        return self._scale_tensor(x_tensor)

    def _compute_edge_index_and_masks(
        self, trans_df: pl.DataFrame, test_size: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Maps edges to node indices and computes chronological train/val/test masks."""
        src_idx = trans_df["From_Acc"].map_elements(
            lambda x: self.account_id_map[x], return_dtype=pl.Int64
        )
        dst_idx = trans_df["To_Acc"].map_elements(
            lambda x: self.account_id_map[x], return_dtype=pl.Int64
        )

        edge_index = torch.stack(
            [
                torch.tensor(src_idx.to_numpy(), dtype=torch.long),
                torch.tensor(dst_idx.to_numpy(), dtype=torch.long),
            ],
            dim=0,
        )

        n_edges = len(trans_df)
        train_cutoff = int(n_edges * (1.0 - test_size))
        val_cutoff = int(n_edges * (1.0 - test_size / 2))

        train_mask = torch.zeros(n_edges, dtype=torch.bool)
        val_mask = torch.zeros(n_edges, dtype=torch.bool)
        test_mask = torch.zeros(n_edges, dtype=torch.bool)

        train_mask[:train_cutoff] = True
        val_mask[train_cutoff:val_cutoff] = True
        test_mask[val_cutoff:] = True

        return edge_index, train_mask, val_mask, test_mask

    def _scale_tensor(self, feature_tensor: torch.Tensor) -> torch.Tensor:
        """Normaliza un tensor usando StandardScaler a media 0 y desviación 1."""
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(feature_tensor.numpy())
        return torch.tensor(scaled_array, dtype=torch.float)

    def build_graph(self, dataset_dir: str, prefix: str, test_size: float = 0.4) -> Data:
        """Construye y devuelve un objeto Data de PyTorch Geometric."""
        accounts_path = self._find_csv(dataset_dir, prefix, "accounts")
        trans_path = self._find_csv(dataset_dir, prefix, "Trans")

        accounts_df = pl.read_csv(str(accounts_path))
        trans_df = pl.read_csv(str(trans_path))

        if "Account_duplicated_0" in trans_df.columns:
            trans_df = trans_df.rename({"Account_duplicated_0": "Account.1"})

        accounts_df, trans_df = self._prepare_accounts_and_transactions(accounts_df, trans_df)

        n_edges = len(trans_df)
        train_cutoff = int(n_edges * (1.0 - test_size))

        x = self._compute_node_features(accounts_df, trans_df, train_cutoff)
        edge_index, train_mask, val_mask, test_mask = self._compute_edge_index_and_masks(
            trans_df, test_size
        )

        trans_df = trans_df.with_columns(
            [pl.col("Amount Received").log1p(), pl.col("Amount Paid").log1p()]
        )

        edge_cols = ["Receiving Currency", "Payment Currency", "Payment Format"]
        drop_cols = [
            "From Bank",
            "Account",
            "To Bank",
            "Account.1",
            "Is Laundering",
            "From_Acc",
            "To_Acc",
        ]
        edge_features_df = trans_df.drop(drop_cols)

        edge_attr = self._encode_dataframe(edge_features_df, edge_cols, self.edge_encoders)
        edge_attr_scaled = self._scale_tensor(edge_attr)

        y = torch.tensor(trans_df["Is Laundering"].to_numpy(), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_scaled,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
