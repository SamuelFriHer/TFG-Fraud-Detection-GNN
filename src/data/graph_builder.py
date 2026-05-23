"""Módulo encargado de la construcción de grafos a partir de los datos tabulares AML."""

from pathlib import Path

import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder  # type: ignore
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
                        # Convert to timestamp float
                        dt = df[col].str.strptime(pl.Datetime, "%Y/%m/%d %H:%M", strict=False)
                        encoded_dict[col] = dt.cast(pl.Float32)
                    else:
                        raise ValueError(f"Columna string no manejada: {col}")
                else:
                    encoded_dict[col] = df[col].cast(pl.Float32)

        encoded_df = pl.DataFrame(encoded_dict)
        return torch.tensor(encoded_df.to_numpy(), dtype=torch.float)

    def _map_account_ids(
        self, accounts_df: pl.DataFrame, trans_df: pl.DataFrame
    ) -> tuple[torch.Tensor, pl.Series, pl.Series]:
        """Mapea los identificadores alfanuméricos de cuenta a índices enteros."""
        # El ID único es la combinación de Bank y Account
        account_series = (
            accounts_df["Bank ID"].cast(pl.String) + "_" + accounts_df["Account Number"]
        )

        # Mapeo a enteros secuenciales
        unique_accounts = account_series.unique().to_list()
        self.account_id_map = {acc: idx for idx, acc in enumerate(unique_accounts)}

        src_accounts = trans_df["From Bank"].cast(pl.String) + "_" + trans_df["Account"]
        dst_accounts = (
            trans_df["To Bank"].cast(pl.String) + "_" + trans_df["Account.1"]
        )  # Fix column name for second Account

        # Mapear, omitir si no existen (aunque deberían)
        src_idx = src_accounts.map_elements(
            lambda x: self.account_id_map.get(x, -1), return_dtype=pl.Int64
        )
        dst_idx = dst_accounts.map_elements(
            lambda x: self.account_id_map.get(x, -1), return_dtype=pl.Int64
        )

        # Filtrar edges válidos
        valid_mask = (src_idx != -1) & (dst_idx != -1)
        src_idx_valid = src_idx.filter(valid_mask)
        dst_idx_valid = dst_idx.filter(valid_mask)

        edge_index = torch.stack(
            [
                torch.tensor(src_idx_valid.to_numpy(), dtype=torch.long),
                torch.tensor(dst_idx_valid.to_numpy(), dtype=torch.long),
            ],
            dim=0,
        )

        return edge_index, valid_mask, account_series

    def build_graph(self, dataset_dir: str, prefix: str) -> Data:
        """Construye y devuelve un objeto Data de PyTorch Geometric."""
        accounts_path = self._find_csv(dataset_dir, prefix, "accounts")
        trans_path = self._find_csv(dataset_dir, prefix, "Trans")

        accounts_df = pl.read_csv(str(accounts_path))
        trans_df = pl.read_csv(str(trans_path))

        # Handle duplicate "Account" columns renamed by Polars
        if "Account_duplicated_0" in trans_df.columns:
            trans_df = trans_df.rename({"Account_duplicated_0": "Account.1"})

        edge_index, valid_mask, _ = self._map_account_ids(accounts_df, trans_df)

        trans_df_valid = trans_df.filter(valid_mask)

        # Procesar nodos
        node_cols = ["Bank Name", "Entity ID", "Entity Name"]
        # Ignoramos ID para features
        node_features_df = accounts_df.select(node_cols)
        x = self._encode_dataframe(node_features_df, node_cols, self.node_encoders)

        # Procesar aristas
        edge_cols = ["Receiving Currency", "Payment Currency", "Payment Format"]
        drop_cols = ["From Bank", "Account", "To Bank", "Account.1", "Is Laundering"]
        edge_features_df = trans_df_valid.drop(drop_cols)

        edge_attr = self._encode_dataframe(edge_features_df, edge_cols, self.edge_encoders)

        y = torch.tensor(trans_df_valid["Is Laundering"].to_numpy(), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
