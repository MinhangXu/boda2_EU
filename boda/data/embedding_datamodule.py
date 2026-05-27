import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ..common import utils


def _coerce_columns(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out = []
    for item in value:
        if isinstance(item, (list, tuple)):
            out.extend(str(x) for x in item)
        else:
            out.append(str(item))
    return out


def _read_rows_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                f"Reading {path} requires a parquet engine such as pyarrow or fastparquet. "
                "Install one, or pass --rows_file rows.csv."
            ) from exc
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported rows table suffix for {path}")


class EmbeddingRegressionDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, weight_tensor=None):
        self.x_tensor = x_tensor.float()
        self.y_tensor = y_tensor.float()
        self.weight_tensor = None if weight_tensor is None else weight_tensor.float()

    def __len__(self):
        return int(self.x_tensor.shape[0])

    def __getitem__(self, idx):
        if self.weight_tensor is None:
            return self.x_tensor[idx], self.y_tensor[idx]
        return self.x_tensor[idx], self.y_tensor[idx], self.weight_tensor[idx]


class EmbeddingRegressionDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("Data Module args")
        group.add_argument("--embedding_dir", type=str, required=True)
        group.add_argument("--embedding_file", type=str, required=True)
        group.add_argument("--rows_file", type=str, default="rows.parquet")
        group.add_argument("--id_column", type=str, default="construct_id")
        group.add_argument("--target_columns", type=str, nargs="+", required=True)
        group.add_argument("--split_column", type=str, default="split")
        group.add_argument("--train_split_value", type=str, default="train")
        group.add_argument("--val_split_value", type=str, default="val")
        group.add_argument("--test_split_value", type=str, default="test")
        group.add_argument("--batch_size", type=int, default=128)
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--pin_memory", type=utils.str2bool, default=True)
        group.add_argument("--standardize_x", type=utils.str2bool, default=True)
        group.add_argument("--standardize_y", type=utils.str2bool, default=True)
        group.add_argument("--drop_missing_targets", type=utils.str2bool, default=True)
        group.add_argument("--use_weights", type=utils.str2bool, default=False)
        group.add_argument("--weight_column", type=str, default="n_barcodes")
        group.add_argument("--min_weight", type=float, default=0.1)
        group.add_argument("--b_cap", type=float, default=10.0)
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        data_args = grouped_args["Data Module args"]
        data_args.target_columns = _coerce_columns(data_args.target_columns)
        return data_args

    def __init__(
        self,
        embedding_dir,
        embedding_file,
        rows_file="rows.parquet",
        id_column="construct_id",
        target_columns=None,
        split_column="split",
        train_split_value="train",
        val_split_value="val",
        test_split_value="test",
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        standardize_x=True,
        standardize_y=True,
        drop_missing_targets=True,
        use_weights=False,
        weight_column="n_barcodes",
        min_weight=0.1,
        b_cap=10.0,
        **kwargs,
    ):
        super().__init__()
        self.embedding_dir = Path(embedding_dir)
        self.embedding_file = embedding_file
        self.rows_file = rows_file
        self.id_column = id_column
        self.target_columns = _coerce_columns(target_columns)
        self.split_column = split_column
        self.train_split_value = train_split_value
        self.val_split_value = val_split_value
        self.test_split_value = test_split_value
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.standardize_x = standardize_x
        self.standardize_y = standardize_y
        self.drop_missing_targets = drop_missing_targets
        self.use_weights = use_weights
        self.weight_column = weight_column
        self.min_weight = min_weight
        self.b_cap = b_cap

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.input_dim = None
        self.n_outputs = len(self.target_columns)
        self.embedding_metadata = None
        self.split_summary = None
        self.df_train = None
        self.df_val = None
        self.df_test = None

    @property
    def rows_path(self):
        path = Path(self.rows_file)
        if path.is_absolute():
            return path
        return self.embedding_dir / path

    @property
    def embedding_path(self):
        path = Path(self.embedding_file)
        if path.is_absolute():
            return path
        return self.embedding_dir / path

    def _load_embedding_payload(self):
        payload = torch.load(self.embedding_path, map_location="cpu")
        if not isinstance(payload, dict) or "embedding" not in payload:
            raise ValueError(f"{self.embedding_path} must contain a dict with an 'embedding' tensor")
        embeddings = payload["embedding"].detach().cpu().float()
        if embeddings.dim() != 2:
            raise ValueError(f"Expected a 2D embedding tensor, got shape {tuple(embeddings.shape)}")
        construct_ids = payload.get(self.id_column, payload.get("construct_id"))
        if construct_ids is None:
            raise ValueError(f"{self.embedding_path} is missing construct_id metadata")
        if len(construct_ids) != embeddings.shape[0]:
            raise ValueError("Embedding construct_id metadata length does not match embedding rows")
        return embeddings, [str(x) for x in construct_ids], payload

    def _align_rows_and_embeddings(self, rows, embeddings, embedding_ids):
        if self.id_column not in rows.columns:
            raise ValueError(f"Rows table is missing id column '{self.id_column}'")
        if self.split_column not in rows.columns:
            raise ValueError(f"Rows table is missing split column '{self.split_column}'")
        missing_targets = [col for col in self.target_columns if col not in rows.columns]
        if missing_targets:
            raise ValueError(f"Rows table is missing target columns: {missing_targets}")

        rows = rows.copy()
        rows[self.id_column] = rows[self.id_column].astype(str)
        if rows[self.id_column].duplicated().any():
            dupes = rows.loc[rows[self.id_column].duplicated(), self.id_column].head(5).tolist()
            raise ValueError(f"Rows table contains duplicate {self.id_column} values, e.g. {dupes}")
        if len(set(embedding_ids)) != len(embedding_ids):
            raise ValueError("Embedding payload contains duplicate construct_id values")

        id_to_idx = {construct_id: i for i, construct_id in enumerate(embedding_ids)}
        missing = [construct_id for construct_id in rows[self.id_column] if construct_id not in id_to_idx]
        if missing:
            raise ValueError(f"Rows table contains ids absent from embeddings, e.g. {missing[:5]}")

        order = torch.tensor([id_to_idx[construct_id] for construct_id in rows[self.id_column]], dtype=torch.long)
        return rows.reset_index(drop=True), embeddings.index_select(0, order)

    def _filter_targets(self, rows, embeddings):
        for col in self.target_columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
        finite_mask = np.isfinite(rows[self.target_columns].to_numpy(dtype=np.float64)).all(axis=1)
        if finite_mask.all():
            return rows, embeddings
        if not self.drop_missing_targets:
            raise ValueError("Target columns contain missing/non-finite values and drop_missing_targets is false")
        keep = torch.tensor(finite_mask, dtype=torch.bool)
        return rows.loc[finite_mask].reset_index(drop=True), embeddings[keep]

    def _validate_split_overlap(self, rows):
        split_sets = {}
        for split_value in (self.train_split_value, self.val_split_value, self.test_split_value):
            ids = set(rows.loc[rows[self.split_column] == split_value, self.id_column].astype(str))
            split_sets[split_value] = ids
        pairs = [
            (self.train_split_value, self.val_split_value),
            (self.train_split_value, self.test_split_value),
            (self.val_split_value, self.test_split_value),
        ]
        for left, right in pairs:
            overlap = split_sets[left] & split_sets[right]
            if overlap:
                raise ValueError(f"Split overlap between {left} and {right}, e.g. {sorted(overlap)[:5]}")

    def _barcode_weight(self, value):
        if not np.isfinite(value):
            return self.min_weight
        raw = math.log1p(float(value)) / math.log1p(float(self.b_cap))
        return float(max(self.min_weight, min(1.0, raw)))

    def _fit_standardizers(self, x_train, y_train):
        self.x_mean = x_train.mean(dim=0, keepdim=True)
        self.x_std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        self.y_mean = y_train.mean(dim=0, keepdim=True)
        self.y_std = y_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)

    def _transform_x(self, x):
        if not self.standardize_x:
            return x
        return (x - self.x_mean) / self.x_std

    def _transform_y(self, y):
        if not self.standardize_y:
            return y
        return (y - self.y_mean) / self.y_std

    def _make_dataset(self, rows, embeddings):
        y = torch.tensor(rows[self.target_columns].to_numpy(dtype=np.float32))
        x = self._transform_x(embeddings)
        y = self._transform_y(y)
        weights = None
        if self.use_weights:
            if self.weight_column not in rows.columns:
                raise ValueError(f"use_weights=true but rows table is missing '{self.weight_column}'")
            raw = pd.to_numeric(rows[self.weight_column], errors="coerce").to_numpy(dtype=np.float64)
            weights = torch.tensor([self._barcode_weight(value) for value in raw], dtype=torch.float32)
        return EmbeddingRegressionDataset(x, y, weights)

    def setup(self, stage=None):
        rows = _read_rows_table(self.rows_path)
        embeddings, embedding_ids, payload = self._load_embedding_payload()
        rows, embeddings = self._align_rows_and_embeddings(rows, embeddings, embedding_ids)
        rows, embeddings = self._filter_targets(rows, embeddings)
        self._validate_split_overlap(rows)

        train_mask = rows[self.split_column] == self.train_split_value
        val_mask = rows[self.split_column] == self.val_split_value
        test_mask = rows[self.split_column] == self.test_split_value
        if not train_mask.any():
            raise ValueError(f"No rows found for train split value '{self.train_split_value}'")
        if not val_mask.any():
            raise ValueError(f"No rows found for validation split value '{self.val_split_value}'")

        x_train = embeddings[torch.tensor(train_mask.to_numpy(), dtype=torch.bool)]
        y_train = torch.tensor(rows.loc[train_mask, self.target_columns].to_numpy(dtype=np.float32))
        self._fit_standardizers(x_train, y_train)

        self.df_train = rows.loc[train_mask].copy().reset_index(drop=True)
        self.df_val = rows.loc[val_mask].copy().reset_index(drop=True)
        self.df_test = rows.loc[test_mask].copy().reset_index(drop=True)

        self.dataset_train = self._make_dataset(self.df_train, embeddings[torch.tensor(train_mask.to_numpy(), dtype=torch.bool)])
        self.dataset_val = self._make_dataset(self.df_val, embeddings[torch.tensor(val_mask.to_numpy(), dtype=torch.bool)])
        self.dataset_test = self._make_dataset(self.df_test, embeddings[torch.tensor(test_mask.to_numpy(), dtype=torch.bool)])

        self.input_dim = int(embeddings.shape[1])
        self.n_outputs = len(self.target_columns)
        self.embedding_metadata = {
            key: value for key, value in payload.items()
            if key != "embedding" and not torch.is_tensor(value)
        }
        self.split_summary = {
            "n_train": len(self.dataset_train),
            "n_val": len(self.dataset_val),
            "n_test": len(self.dataset_test),
            "input_dim": self.input_dim,
            "n_outputs": self.n_outputs,
        }

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
