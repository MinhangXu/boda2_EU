import argparse
import hashlib
from functools import partial

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ..common import constants, utils


class DNARegressionDataset(Dataset):
    def __init__(self, dna_tensor, target_tensor, weight_tensor=None, use_reverse_complements=False):
        self.dna_tensor = dna_tensor
        self.target_tensor = target_tensor
        self.weight_tensor = weight_tensor
        self.use_reverse_complements = use_reverse_complements
        self.n_examples = int(dna_tensor.shape[0])

    def __len__(self):
        return self.n_examples * 2 if self.use_reverse_complements else self.n_examples

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(idx)
        take_rc = self.use_reverse_complements and (idx % 2 == 1)
        item_idx = (idx // 2) if self.use_reverse_complements else idx
        dna = self.dna_tensor[item_idx]
        target = self.target_tensor[item_idx]
        if take_rc:
            dna = utils.reverse_complement_onehot(dna)
        if self.weight_tensor is None:
            return dna, target
        return dna, target, self.weight_tensor[item_idx]


class BashorDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Data Module args')
        group.add_argument('--datafile_path', type=str, required=True)
        group.add_argument('--sep', type=str, choices={'space', 'tab', 'comma', ' ', '\t', ','}, default='\t')
        group.add_argument('--sequence_column', type=str, default='Enhancers')
        group.add_argument('--target_column', type=str, default='RNA_DNA_Ratio_log10_scaled')
        group.add_argument('--barcode_column', type=str, default='n_barcodes')
        group.add_argument('--batch_size', type=int, default=128)
        group.add_argument('--padded_seq_len', type=int, default=600)
        group.add_argument('--left_flank', type=str, default=constants.MPRA_UPSTREAM)
        group.add_argument('--right_flank', type=str, default=constants.MPRA_DOWNSTREAM)
        group.add_argument(
            '--padding_mode',
            type=str,
            default='mpra_flank',
            choices=['mpra_flank', 'neutral', 'none'],
            help=(
                "'mpra_flank' pads with the BODA/Malinois MPRA context, "
                "'neutral' pads with neutral bases such as N, and 'none' uses "
                "raw sequences and requires equal lengths."
            ),
        )
        group.add_argument('--neutral_pad_char', type=str, default='N')
        group.add_argument('--num_workers', type=int, default=8)
        group.add_argument('--normalize', type=utils.str2bool, default=True)
        group.add_argument('--split_seed', type=int, default=7)
        group.add_argument('--test_min_barcodes', type=int, default=4)
        group.add_argument('--train_min_barcodes', type=int, default=1)
        group.add_argument('--train_max_barcodes', type=int, default=None)
        group.add_argument('--val_frac_within_hq', type=float, default=0.2)
        group.add_argument('--test_frac_within_hq', type=float, default=0.2)
        group.add_argument('--val_size_within_hq', type=int, default=None)
        group.add_argument('--test_size_within_hq', type=int, default=None)
        group.add_argument('--train_size_frac', type=float, default=1.0)
        group.add_argument('--train_size_n', type=int, default=None)
        group.add_argument('--min_train_size', type=int, default=32)
        group.add_argument('--train_sampling_mode', type=str, default='hq_first', choices=['hq_first', 'random'])
        group.add_argument('--train_subsample_seed', type=int, default=None)
        group.add_argument('--use_reverse_complements', type=utils.str2bool, default=False)
        group.add_argument('--barcode_weighting', type=utils.str2bool, default=False)
        group.add_argument('--barcode_weight_cap', type=float, default=8.0)
        group.add_argument('--barcode_weight_min', type=float, default=0.1)  # minimum weight for barcode weighting
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        return parser

    @staticmethod
    def process_args(grouped_args):
        data_args = grouped_args['Data Module args']
        data_args.sep = {'space': ' ', 'tab': '\t', 'comma': ',', ' ': ' ', '\t': '\t', ',': ','}[data_args.sep]
        return data_args

    def __init__(self,
                 datafile_path,
                 sep='\t',
                 sequence_column='Enhancers',
                 target_column='RNA_DNA_Ratio_log10_scaled',
                 barcode_column='n_barcodes',
                 batch_size=64,
                 padded_seq_len=600,
                 left_flank=constants.MPRA_UPSTREAM,
                 right_flank=constants.MPRA_DOWNSTREAM,
                 padding_mode='mpra_flank',
                 neutral_pad_char='N',
                 num_workers=8,
                 normalize=True,
                 split_seed=7,
                 test_min_barcodes=4,
                 train_min_barcodes=1,
                 train_max_barcodes=None,
                 val_frac_within_hq=0.2,
                 test_frac_within_hq=0.2,
                 val_size_within_hq=None,
                 test_size_within_hq=None,
                 train_size_frac=1.0,
                 train_size_n=None,
                 min_train_size=32,
                 train_sampling_mode='hq_first',
                 train_subsample_seed=None,
                 use_reverse_complements=False,
                 barcode_weighting=False,
                 barcode_weight_cap=8.0,
                 barcode_weight_min=0.1,
                 **kwargs):
        super().__init__()
        self.datafile_path = datafile_path
        self.sep = sep
        self.sequence_column = sequence_column
        self.target_column = target_column
        self.barcode_column = barcode_column
        self.batch_size = batch_size
        self.padded_seq_len = padded_seq_len
        self.left_flank = left_flank
        self.right_flank = right_flank
        self.padding_mode = padding_mode
        self.neutral_pad_char = neutral_pad_char
        self.num_workers = num_workers
        self.normalize = normalize
        self.split_seed = split_seed
        self.test_min_barcodes = test_min_barcodes
        self.train_min_barcodes = train_min_barcodes
        self.train_max_barcodes = train_max_barcodes
        self.val_frac_within_hq = val_frac_within_hq
        self.test_frac_within_hq = test_frac_within_hq
        self.val_size_within_hq = val_size_within_hq
        self.test_size_within_hq = test_size_within_hq
        self.train_size_frac = train_size_frac
        self.train_size_n = train_size_n
        self.min_train_size = min_train_size
        self.train_sampling_mode = train_sampling_mode
        self.train_subsample_seed = split_seed if train_subsample_seed is None else train_subsample_seed
        self.use_reverse_complements = use_reverse_complements
        self.barcode_weighting = barcode_weighting
        self.barcode_weight_cap = barcode_weight_cap
        self.barcode_weight_min = barcode_weight_min

        if self.padding_mode not in {'mpra_flank', 'neutral', 'none'}:
            raise ValueError(f"Unknown padding_mode: {self.padding_mode}")
        if len(str(self.neutral_pad_char)) != 1:
            raise ValueError("neutral_pad_char must be a single character")
        if self.train_max_barcodes is not None and self.train_max_barcodes < self.train_min_barcodes:
            raise ValueError(
                f"train_max_barcodes={self.train_max_barcodes} is less than "
                f"train_min_barcodes={self.train_min_barcodes}"
            )
        if self.train_size_n is not None and self.train_size_n < self.min_train_size:
            raise ValueError(
                f"train_size_n={self.train_size_n} is smaller than min_train_size={self.min_train_size}"
            )

        self.pad_column_name = 'padded_seq'
        if self.padding_mode == 'mpra_flank':
            self.padding_fn = partial(
                utils.row_pad_sequence,
                in_column_name=self.sequence_column,
                padded_seq_len=self.padded_seq_len,
                upStreamSeq=self.left_flank,
                downStreamSeq=self.right_flank,
            )
        else:
            self.padding_fn = self._pad_sequence_without_flanks

        self.dataset_train = None
        self.dataset_train_eval = None
        self.dataset_val = None
        self.dataset_test = None
        self.target_mean = None
        self.target_std = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.df_train_pool = None
        self.df_train_pool_leftover_hq = None
        self.df_train_pool_lower_quality = None
        self.split_summary = None

    def _barcode_weight(self, n):
        raw = np.log1p(float(n)) / np.log1p(float(self.barcode_weight_cap))
        return float(max(self.barcode_weight_min, min(1.0, raw)))

    def _pad_sequence_without_flanks(self, row):
        sequence = str(row[self.sequence_column]).upper()
        if self.padding_mode == 'none':
            return sequence

        target_len = int(self.padded_seq_len)
        if target_len <= 0:
            return sequence
        if len(sequence) > target_len:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds padded_seq_len={target_len} "
                "for neutral padding."
            )

        padding_len = target_len - len(sequence)
        left_len = padding_len // 2
        right_len = padding_len - left_len
        pad_char = str(self.neutral_pad_char).upper()
        return f"{pad_char * left_len}{sequence}{pad_char * right_len}"

    def _validate_padded_lengths(self, df):
        lengths = df[self.pad_column_name].astype(str).str.len()
        if lengths.nunique() > 1:
            counts = lengths.value_counts().sort_index().head(10).to_dict()
            raise ValueError(
                f"Sequences have variable lengths after padding_mode={self.padding_mode}: "
                f"{counts}. Use padding_mode='neutral' with a fixed padded_seq_len, "
                "or provide equal-length raw sequences for padding_mode='none'."
            )

    def _prep_df(self):
        df = pd.read_csv(self.datafile_path, sep=self.sep).copy()
        df[self.target_column] = pd.to_numeric(df[self.target_column], errors='coerce')
        df[self.barcode_column] = pd.to_numeric(df[self.barcode_column], errors='coerce')
        df = df.loc[
            df[self.sequence_column].notna()
            & df[self.barcode_column].notna()
            & np.isfinite(df[self.target_column])
        ].reset_index(drop=True)
        df['row_id'] = np.arange(len(df))
        df[self.pad_column_name] = df.apply(self.padding_fn, axis=1)
        self._validate_padded_lengths(df)
        return df

    def _build_train_pool_components(self, df_rest):
        train_mask = df_rest[self.barcode_column] >= self.train_min_barcodes
        if self.train_max_barcodes is not None:
            train_mask = train_mask & (df_rest[self.barcode_column] <= self.train_max_barcodes)
        eligible = df_rest.loc[train_mask].copy().reset_index(drop=True)
        if len(eligible) < self.min_train_size:
            raise ValueError(
                f'Train pool too small for barcode range '
                f'[{self.train_min_barcodes}, {self.train_max_barcodes}]: {len(eligible)} rows'
            )
        leftover_hq = eligible.loc[eligible[self.barcode_column] >= self.test_min_barcodes].copy().reset_index(drop=True)
        lower_quality = eligible.loc[eligible[self.barcode_column] < self.test_min_barcodes].copy().reset_index(drop=True)
        return eligible, leftover_hq, lower_quality

    def _stable_row_id_hash(self, df):
        if 'row_id' not in df.columns:
            return None
        row_ids = sorted(int(row_id) for row_id in df['row_id'].to_numpy())
        payload = ','.join(str(row_id) for row_id in row_ids).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _barcode_histogram(self, df):
        counts = df[self.barcode_column].astype(int).value_counts().sort_index()
        return {str(int(barcode)): int(count) for barcode, count in counts.items()}

    def _subsample_train_pool(self, eligible, leftover_hq, lower_quality):
        if self.train_size_n is not None:
            train_n = int(self.train_size_n)
            if train_n > len(eligible):
                raise ValueError(
                    f"Requested train_size_n={train_n}, but only {len(eligible)} rows are "
                    f"available for barcode range [{self.train_min_barcodes}, {self.train_max_barcodes}]"
                )
        elif self.train_size_frac >= 1.0:
            df_train = eligible.copy()
            df_train['train_sampling_mode'] = self.train_sampling_mode
            df_train['is_leftover_hq_train'] = df_train[self.barcode_column] >= self.test_min_barcodes
            return df_train
        else:
            train_n = max(self.min_train_size, int(round(len(eligible) * self.train_size_frac)))
            train_n = min(train_n, len(eligible))

        if self.train_sampling_mode == 'random':
            rng = np.random.default_rng(int(self.train_subsample_seed))
            perm = rng.permutation(len(eligible))
            df_train = eligible.iloc[perm[:train_n]].copy().reset_index(drop=True)
        elif self.train_sampling_mode == 'hq_first':
            hq_take = min(train_n, len(leftover_hq))
            lq_take = max(0, train_n - hq_take)
            sampled_parts = []
            if hq_take > 0:
                sampled_parts.append(leftover_hq.sample(n=hq_take, random_state=self.train_subsample_seed, replace=False))
            if lq_take > 0:
                sampled_parts.append(lower_quality.sample(n=lq_take, random_state=self.train_subsample_seed + 100003, replace=False))
            df_train = pd.concat(sampled_parts, axis=0, ignore_index=True)
            df_train = df_train.sample(frac=1.0, random_state=self.train_subsample_seed + 7919).reset_index(drop=True)
        else:
            raise ValueError(f'Unknown train_sampling_mode: {self.train_sampling_mode}')
        df_train['train_sampling_mode'] = self.train_sampling_mode
        df_train['is_leftover_hq_train'] = df_train[self.barcode_column] >= self.test_min_barcodes
        return df_train

    def _split_df(self, df):
        hq_df = df.loc[df[self.barcode_column] >= self.test_min_barcodes].copy()
        if len(hq_df) < 5:
            raise ValueError('Not enough HQ rows for val/test split.')

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(hq_df.index.to_numpy())
        n_hq = len(perm)
        if self.test_size_within_hq is None:
            n_test = max(1, int(round(n_hq * self.test_frac_within_hq)))
        else:
            n_test = int(self.test_size_within_hq)
        if self.val_size_within_hq is None:
            n_val = max(1, int(round(n_hq * self.val_frac_within_hq)))
        else:
            n_val = int(self.val_size_within_hq)
        if n_test < 1 or n_val < 1:
            raise ValueError(f'Val/test sizes must be positive: val={n_val}, test={n_test}')
        if n_test + n_val >= n_hq:
            if self.test_size_within_hq is not None or self.val_size_within_hq is not None:
                raise ValueError(
                    f'Requested val/test sizes ({n_val}/{n_test}) exhaust HQ rows ({n_hq}).'
                )
            n_test = max(1, n_hq // 5)
            n_val = max(1, n_hq // 5)

        test_idx = set(perm[:n_test].tolist())
        val_idx = set(perm[n_test:n_test + n_val].tolist())
        used_idx = test_idx | val_idx

        df_test = df.loc[df.index.isin(test_idx)].copy().reset_index(drop=True)
        df_val = df.loc[df.index.isin(val_idx)].copy().reset_index(drop=True)
        df_rest = df.loc[~df.index.isin(used_idx)].copy().reset_index(drop=True)

        eligible, leftover_hq, lower_quality = self._build_train_pool_components(df_rest)
        df_train = self._subsample_train_pool(eligible, leftover_hq, lower_quality)

        self.df_train_pool = eligible.copy()
        self.df_train_pool_leftover_hq = leftover_hq.copy()
        self.df_train_pool_lower_quality = lower_quality.copy()
        self.split_summary = {
            'n_total': int(len(df)),
            'n_hq_total': int(len(hq_df)),
            'n_test': int(len(df_test)),
            'n_val': int(len(df_val)),
            'n_train_pool_eligible': int(len(eligible)),
            'n_train_pool_leftover_hq': int(len(leftover_hq)),
            'n_train_pool_lower_quality': int(len(lower_quality)),
            'n_train_final': int(len(df_train)),
            'n_train_final_hq': int((df_train[self.barcode_column] >= self.test_min_barcodes).sum()),
            'n_train_final_lower_quality': int((df_train[self.barcode_column] < self.test_min_barcodes).sum()),
            'train_sampling_mode': self.train_sampling_mode,
            'train_subsample_seed': int(self.train_subsample_seed),
            'train_size_frac': float(self.train_size_frac),
            'train_size_n': None if self.train_size_n is None else int(self.train_size_n),
            'train_min_barcodes': int(self.train_min_barcodes),
            'train_max_barcodes': None if self.train_max_barcodes is None else int(self.train_max_barcodes),
            'test_min_barcodes': int(self.test_min_barcodes),
            'val_size_within_hq': None if self.val_size_within_hq is None else int(self.val_size_within_hq),
            'test_size_within_hq': None if self.test_size_within_hq is None else int(self.test_size_within_hq),
            'train_pool_barcode_histogram': self._barcode_histogram(eligible),
            'train_final_barcode_histogram': self._barcode_histogram(df_train),
            'train_pool_row_id_hash': self._stable_row_id_hash(eligible),
            'train_final_row_id_hash': self._stable_row_id_hash(df_train),
        }
        return df_train, df_val, df_test

    def _standardize_targets(self, df_train, df_val, df_test):
        self.target_mean = float(df_train[self.target_column].mean())
        self.target_std = float(df_train[self.target_column].std())
        if (not np.isfinite(self.target_std)) or self.target_std < 1e-8:
            self.target_std = 1.0
        out = []
        for frame in (df_train, df_val, df_test):
            frame = frame.copy()
            if self.normalize:
                frame['target_processed'] = (frame[self.target_column] - self.target_mean) / self.target_std
            else:
                frame['target_processed'] = frame[self.target_column]
            if self.barcode_weighting:
                frame['sample_weight'] = frame[self.barcode_column].map(self._barcode_weight)
            out.append(frame)
        return out

    def _df_to_dataset(self, df, training=False):
        dna_list = [utils.row_dna2tensor(row, in_column_name=self.pad_column_name) for _, row in df.iterrows()]
        dna_tensor = torch.stack(dna_list, dim=0)
        y = torch.tensor(df['target_processed'].to_numpy(dtype=np.float32)).view(-1, 1)
        w = None
        if 'sample_weight' in df.columns:
            w = torch.tensor(df['sample_weight'].to_numpy(dtype=np.float32))
        return DNARegressionDataset(
            dna_tensor=dna_tensor,
            target_tensor=y,
            weight_tensor=w,
            use_reverse_complements=(training and self.use_reverse_complements),
        )

    def setup(self, stage=None):
        df = self._prep_df()
        df_train, df_val, df_test = self._split_df(df)
        df_train, df_val, df_test = self._standardize_targets(df_train, df_val, df_test)

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.dataset_train = self._df_to_dataset(df_train, training=True)
        self.dataset_train_eval = self._df_to_dataset(df_train, training=False)
        self.dataset_val = self._df_to_dataset(df_val, training=False)
        self.dataset_test = self._df_to_dataset(df_test, training=False)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def train_eval_dataloader(self):
        dataset = self.dataset_train_eval if self.dataset_train_eval is not None else self.dataset_train
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class Lib1EnhancerDataModule(BashorDataModule):
    pass


class Lib1ThreePrimeDataModule(BashorDataModule):
    pass


class Lib1PromoterDataModule(BashorDataModule):
    pass


class Lib1IntronDataModule(BashorDataModule):
    pass


class Lib1FivePrimeDataModule(BashorDataModule):
    pass
