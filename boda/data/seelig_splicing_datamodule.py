import argparse

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..common import utils


class SeeligSplicingScalarDataset(Dataset):
    """One-hot DNA regression dataset with optional train-time RC augmentation."""

    def __init__(self, dna_tensor, target_tensor, use_reverse_complements=False):
        self.dna_tensor = dna_tensor
        self.target_tensor = target_tensor
        self.use_reverse_complements = use_reverse_complements
        self.n_examples = int(dna_tensor.shape[0])

    def __len__(self):
        return self.n_examples * 2 if self.use_reverse_complements else self.n_examples

    def __getitem__(self, idx):
        take_rc = self.use_reverse_complements and (idx % 2 == 1)
        item_idx = (idx // 2) if self.use_reverse_complements else idx
        dna = self.dna_tensor[item_idx]
        if take_rc:
            dna = utils.reverse_complement_onehot(dna)
        return dna, self.target_tensor[item_idx]


class SeeligA5SSScalarDataModule(pl.LightningDataModule):
    """
    BODA DataModule for the Rosenberg/Seelig 2015 A5SS SD1 scalar task.

    Expected input is the CSV from `src/learn/prepare_seelig_splicing_dataset.py`
    with one row per A5SS sequence and a `p_sd1` target.
    """

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("Seelig A5SS Scalar DataModule")
        group.add_argument("--datafile_path", type=str, required=True)
        group.add_argument("--sequence_column", type=str, default="seq")
        group.add_argument("--target_column", type=str, default="p_sd1")
        group.add_argument("--fold_column", type=str, default="fold")
        group.add_argument("--read_count_column", type=str, default="read_count_total")
        group.add_argument("--min_read_count", type=int, default=1)

        group.add_argument("--train_split", type=float, default=0.8)
        group.add_argument("--val_split", type=float, default=0.1)
        group.add_argument("--test_split", type=float, default=0.1)
        group.add_argument("--split_by_fold", type=utils.str2bool, default=True)

        group.add_argument("--batch_size", type=int, default=512)
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--normalize_target", type=utils.str2bool, default=False)
        group.add_argument("--use_reverse_complements", type=utils.str2bool, default=False)
        group.add_argument("--pin_memory", type=utils.str2bool, default=True)
        group.add_argument("--seed", type=int, default=42)
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        return grouped_args["Seelig A5SS Scalar DataModule"]

    def __init__(
        self,
        datafile_path,
        sequence_column="seq",
        target_column="p_sd1",
        fold_column="fold",
        read_count_column="read_count_total",
        min_read_count=1,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        split_by_fold=True,
        batch_size=512,
        num_workers=8,
        normalize_target=False,
        use_reverse_complements=False,
        pin_memory=True,
        seed=42,
        **kwargs,
    ):
        super().__init__()
        self.datafile_path = datafile_path
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.fold_column = fold_column
        self.read_count_column = read_count_column
        self.min_read_count = min_read_count
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.split_by_fold = split_by_fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize_target = normalize_target
        self.use_reverse_complements = use_reverse_complements
        self.pin_memory = pin_memory
        self.seed = seed

        self.target_mean = None
        self.target_std = None
        self.seq_len = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.split_summary = None
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def _load_df(self):
        df = pd.read_csv(self.datafile_path)
        required = [self.sequence_column, self.target_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.datafile_path}: {missing}")

        df = df.dropna(subset=required).copy()
        df[self.sequence_column] = df[self.sequence_column].astype(str).str.upper()
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce")
        df = df[np.isfinite(df[self.target_column])].copy()

        if self.read_count_column and self.read_count_column in df.columns:
            df[self.read_count_column] = pd.to_numeric(df[self.read_count_column], errors="coerce")
            df = df[df[self.read_count_column] >= self.min_read_count].copy()

        lengths = df[self.sequence_column].str.len()
        if lengths.nunique() != 1:
            counts = lengths.value_counts().sort_index().head(10).to_dict()
            raise ValueError(f"Expected fixed-length A5SS sequences, observed lengths: {counts}")
        self.seq_len = int(lengths.iloc[0])
        return df.reset_index(drop=True)

    def _split_df(self, df):
        if self.split_by_fold and self.fold_column in df.columns:
            df_train = df[df[self.fold_column] == "train"].copy()
            df_val = df[df[self.fold_column] == "val"].copy()
            df_test = df[df[self.fold_column] == "test"].copy()
        else:
            if self.train_split <= 0 or self.val_split < 0 or self.test_split < 0:
                raise ValueError("train_split must be positive and val/test splits nonnegative.")
            if self.train_split + self.val_split + self.test_split <= 0:
                raise ValueError("At least one split fraction must be positive.")
            scale = self.train_split + self.val_split + self.test_split
            train_frac = self.train_split / scale
            val_frac = self.val_split / scale

            shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
            n_total = len(shuffled)
            n_train = int(round(n_total * train_frac))
            n_val = int(round(n_total * val_frac))
            n_train = min(max(1, n_train), n_total - 2)
            n_val = min(max(1, n_val), n_total - n_train - 1)
            df_train = shuffled.iloc[:n_train].copy()
            df_val = shuffled.iloc[n_train:n_train + n_val].copy()
            df_test = shuffled.iloc[n_train + n_val:].copy()

        for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if len(split_df) == 0:
                raise ValueError(f"{split_name} split is empty for {self.datafile_path}")
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

    def _normalize_splits(self, df_train, df_val, df_test):
        if not self.normalize_target:
            return df_train, df_val, df_test

        self.target_mean = float(df_train[self.target_column].mean())
        self.target_std = float(df_train[self.target_column].std())
        if not np.isfinite(self.target_std) or self.target_std < 1e-8:
            self.target_std = 1.0

        out = []
        for frame in (df_train, df_val, df_test):
            frame = frame.copy()
            frame[self.target_column] = (frame[self.target_column] - self.target_mean) / self.target_std
            out.append(frame)
        return out

    def _df_to_dataset(self, df, training=False):
        sequences = [utils.dna2tensor(seq) for seq in df[self.sequence_column]]
        dna_tensor = torch.stack(sequences, dim=0)
        target_tensor = torch.tensor(
            df[self.target_column].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ).view(-1, 1)

        if training:
            return SeeligSplicingScalarDataset(
                dna_tensor,
                target_tensor,
                use_reverse_complements=self.use_reverse_complements,
            )
        return TensorDataset(dna_tensor, target_tensor)

    def setup(self, stage=None):
        df = self._load_df()
        df_train, df_val, df_test = self._split_df(df)
        df_train, df_val, df_test = self._normalize_splits(df_train, df_val, df_test)

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.split_summary = {
            "n_train": int(len(df_train)),
            "n_val": int(len(df_val)),
            "n_test": int(len(df_test)),
            "seq_len": int(self.seq_len),
            "target_column": self.target_column,
            "normalize_target": bool(self.normalize_target),
        }

        self.dataset_train = self._df_to_dataset(df_train, training=True)
        self.dataset_val = self._df_to_dataset(df_val, training=False)
        self.dataset_test = self._df_to_dataset(df_test, training=False)

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
