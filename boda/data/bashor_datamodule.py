import argparse
import ast
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from ..common import constants, utils


LIB1_SPLIT_SCHEMA_VERSION = 'lib1_dedup_split_v1'
LIB1_SPLIT_ID_HASH_ALGORITHM = 'sha256_canonical_sorted_json_v1'


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _stable_id_hash(values):
    return _canonical_json_sha256(sorted(str(value) for value in values))


def _split_assignment_hash(assignments):
    membership = sorted(
        (
            {
                'construct_id': str(row['construct_id']),
                'partition': str(row['partition']),
                'development_fold': (
                    None
                    if row.get('development_fold') is None
                    else int(row['development_fold'])
                ),
            }
            for row in assignments
        ),
        key=lambda row: row['construct_id'],
    )
    return _canonical_json_sha256(membership)


def _coerce_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {'none', 'null'}:
            return []
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except Exception:
                pass
        return value.split()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_coerce_string_list(item))
        return out
    return [str(value)]


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
        group.add_argument(
            '--split_manifest_path',
            type=str,
            default=None,
            help=(
                'Optional frozen stable-ID split manifest. When supplied, audit '
                'rows are excluded and no test loader is created.'
            ),
        )
        group.add_argument(
            '--manifest_mode',
            type=str,
            default='development',
            choices=[
                'development',
                'development_inner_oof',
                'final_refit',
                'audit_eval',
            ],
            help=(
                "How to consume a frozen split manifest. 'development' uses one "
                "development fold for validation and exposes no test loader; "
                "'development_inner_oof' physically excludes frozen final-test "
                "rows, uses split_fold as an untouched outer OOF fold, uses the "
                "next development fold for checkpoint selection, and trains on "
                "train_only plus the other three folds; "
                "'final_refit' trains on every eligible non-audit row with no "
                "validation or test loader; 'audit_eval' preserves the same "
                "all-development normalization reference and exposes only the "
                "frozen audit rows as the test loader."
            ),
        )
        group.add_argument(
            '--split_fold',
            type=int,
            default=None,
            help=(
                'Development fold selected from split_manifest_path. Falls back '
                'to the trainer --development_fold, then fold 0.'
            ),
        )
        group.add_argument(
            '--split_id_column',
            type=str,
            default=None,
            help='Stable dataset ID column; defaults to the manifest dataset.id_column.',
        )
        group.add_argument(
            '--expected_data_sha256',
            '--expected_dataset_sha256',
            dest='expected_data_sha256',
            type=str,
            default=None,
            help='Optional expected SHA256 for datafile_path.',
        )
        group.add_argument(
            '--expected_split_sha256',
            '--expected_split_manifest_sha256',
            dest='expected_split_sha256',
            type=str,
            default=None,
            help='Optional expected SHA256 for split_manifest_path.',
        )
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
        main_args = grouped_args.get('Main args')
        development_fold = (
            None if main_args is None else getattr(main_args, 'development_fold', None)
        )
        if data_args.split_fold is None:
            data_args.split_fold = 0 if development_fold is None else int(development_fold)
        elif development_fold is not None and int(data_args.split_fold) != int(development_fold):
            raise ValueError(
                f'Conflicting --split_fold={data_args.split_fold} and '
                f'--development_fold={development_fold}'
            )
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
                 split_manifest_path=None,
                 manifest_mode='development',
                 split_fold=0,
                 split_id_column=None,
                 expected_data_sha256=None,
                 expected_split_sha256=None,
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
        self.split_manifest_path = (
            None
            if split_manifest_path is None or str(split_manifest_path).strip() in {'', 'None', 'null'}
            else str(split_manifest_path)
        )
        self.manifest_mode = str(manifest_mode).strip().lower()
        if self.manifest_mode not in {
            'development', 'development_inner_oof', 'final_refit', 'audit_eval'
        }:
            raise ValueError(f'Unknown manifest_mode={self.manifest_mode!r}')
        self.split_fold = 0 if split_fold is None else int(split_fold)
        self.split_id_column = (
            None
            if split_id_column is None or str(split_id_column).strip() in {'', 'None', 'null'}
            else str(split_id_column)
        )
        self.expected_data_sha256 = (
            None
            if expected_data_sha256 is None or str(expected_data_sha256).strip() == ''
            else str(expected_data_sha256).strip().lower()
        )
        self.expected_split_sha256 = (
            None
            if expected_split_sha256 is None or str(expected_split_sha256).strip() == ''
            else str(expected_split_sha256).strip().lower()
        )
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
        if not np.isfinite(float(self.barcode_weight_cap)) or float(self.barcode_weight_cap) <= 0:
            raise ValueError("barcode_weight_cap must be finite and greater than zero")
        if (
            not np.isfinite(float(self.barcode_weight_min))
            or float(self.barcode_weight_min) <= 0
            or float(self.barcode_weight_min) > 1
        ):
            raise ValueError("barcode_weight_min must be finite and in (0, 1]")
        for label, expected_sha in (
            ('expected_data_sha256', self.expected_data_sha256),
            ('expected_split_sha256', self.expected_split_sha256),
        ):
            if expected_sha is not None and (
                len(expected_sha) != 64
                or any(character not in '0123456789abcdef' for character in expected_sha)
            ):
                raise ValueError(f'{label} must be a lowercase 64-character SHA256')
        if self.expected_split_sha256 is not None and self.split_manifest_path is None:
            raise ValueError(
                'expected_split_sha256 requires split_manifest_path'
            )
        if self.manifest_mode != 'development' and self.split_manifest_path is None:
            raise ValueError(
                f"manifest_mode={self.manifest_mode!r} requires split_manifest_path"
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
        self.dataset_oof = None
        self.target_mean = None
        self.target_std = None
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.df_oof = None
        self.df_audit = None
        self.df_train_pool = None
        self.df_train_pool_leftover_hq = None
        self.df_train_pool_lower_quality = None
        self.split_summary = None
        self.split_manifest = None
        self.datafile_sha256 = None
        self.split_manifest_sha256 = None

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
        read_kwargs = {}
        if self.split_manifest is not None and self.manifest_mode in {
            'development_inner_oof', 'final_refit'
        }:
            # Exclude audit rows before pandas parses any sequence, target, or
            # barcode field.  Reading only the stable-ID column is sufficient
            # to turn the frozen manifest assignment into physical CSV row
            # skips; the training process never retains audit content.
            id_column = str(self.split_manifest['dataset']['id_column'])
            id_frame = pd.read_csv(
                self.datafile_path, sep=self.sep, usecols=[id_column]
            )
            audit_ids = {
                str(row['construct_id'])
                for row in self.split_manifest['assignments']
                if row['partition'] == 'audit_test'
            }
            skiprows = [
                index + 1
                for index, stable_id in enumerate(id_frame[id_column].astype(str))
                if stable_id in audit_ids
            ]
            if len(skiprows) != int(
                self.split_manifest['expected_counts']['audit_test']
            ):
                raise ValueError('Could not resolve every frozen audit row for exclusion')
            read_kwargs['skiprows'] = skiprows
        df = pd.read_csv(self.datafile_path, sep=self.sep, **read_kwargs).copy()
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

    def _validate_datafile_hash(self):
        self.datafile_sha256 = _sha256_file(self.datafile_path)
        if (
            self.expected_data_sha256 is not None
            and self.datafile_sha256 != self.expected_data_sha256
        ):
            raise ValueError(
                f'Dataset SHA256 mismatch for {self.datafile_path}: expected '
                f'{self.expected_data_sha256}, observed {self.datafile_sha256}'
            )

    def _load_split_manifest(self):
        if self.split_manifest_path is None:
            return None
        path = Path(self.split_manifest_path)
        self.split_manifest_sha256 = _sha256_file(path)
        if (
            self.expected_split_sha256 is not None
            and self.split_manifest_sha256 != self.expected_split_sha256
        ):
            raise ValueError(
                f'Split manifest SHA256 mismatch for {path}: expected '
                f'{self.expected_split_sha256}, observed {self.split_manifest_sha256}'
            )
        with path.open() as handle:
            manifest = json.load(handle)
        if manifest.get('schema_version') != LIB1_SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported split schema {manifest.get('schema_version')!r}; "
                f'expected {LIB1_SPLIT_SCHEMA_VERSION!r}'
            )
        dataset = manifest.get('dataset')
        if not isinstance(dataset, dict):
            raise ValueError('Split manifest is missing dataset metadata')
        expected_dataset_sha = str(dataset.get('sha256', '')).lower()
        if expected_dataset_sha != self.datafile_sha256:
            raise ValueError(
                f'Split manifest dataset SHA256 mismatch: manifest binds '
                f'{expected_dataset_sha}, datafile is {self.datafile_sha256}'
            )
        expected_rows = int(dataset.get('row_count', -1))
        if expected_rows < 0:
            raise ValueError('Split manifest dataset.row_count is missing')
        manifest_id_column = dataset.get('id_column')
        if not manifest_id_column:
            raise ValueError('Split manifest dataset.id_column is missing')
        if self.split_id_column is None:
            self.split_id_column = str(manifest_id_column)
        elif self.split_id_column != str(manifest_id_column):
            raise ValueError(
                f'split_id_column={self.split_id_column!r} does not match '
                f'manifest dataset.id_column={manifest_id_column!r}'
            )
        # A frozen split is also a binding data/model-input contract.  The
        # learn-ready files intentionally retain legacy target columns, so a
        # hash check alone cannot prevent a malformed command from selecting
        # the wrong target or padding policy.
        runtime_contract = {
            'sequence_column': self.sequence_column,
            'target_column': self.target_column,
            'barcode_column': self.barcode_column,
            'padded_seq_len': int(self.padded_seq_len),
            'padding_mode': self.padding_mode,
            'neutral_pad_char': str(self.neutral_pad_char),
            'normalize': bool(self.normalize),
        }
        for field, observed in runtime_contract.items():
            if field not in dataset:
                raise ValueError(
                    f'Split manifest dataset metadata is missing required '
                    f'runtime contract field {field!r}'
                )
            expected = dataset[field]
            if expected != observed:
                raise ValueError(
                    f'Runtime {field}={observed!r} does not match frozen split '
                    f'manifest dataset.{field}={expected!r}'
                )
        if self.padding_mode == 'mpra_flank':
            for field, sequence in (
                ('left_flank_sha256', self.left_flank),
                ('right_flank_sha256', self.right_flank),
            ):
                expected = str(dataset.get(field, '')).lower()
                if len(expected) != 64:
                    raise ValueError(
                        f'Split manifest dataset.{field} must freeze the MPRA flank hash'
                    )
                observed = hashlib.sha256(str(sequence).encode('utf-8')).hexdigest()
                if observed != expected:
                    raise ValueError(
                        f'Runtime {field}={observed!r} does not match frozen split '
                        f'manifest dataset.{field}={expected!r}'
                    )
        n_folds = int(manifest.get('n_development_folds', -1))
        if n_folds < 2:
            raise ValueError('Split manifest must contain at least two development folds')
        if self.split_fold < 0 or self.split_fold >= n_folds:
            raise ValueError(
                f'split_fold={self.split_fold} is outside [0, {n_folds - 1}]'
            )
        threshold = int(
            manifest.get('policy', {}).get(
                'heldout_min_barcodes', dataset.get('high_barcode_threshold', -1)
            )
        )
        if threshold != int(self.test_min_barcodes):
            raise ValueError(
                f'test_min_barcodes={self.test_min_barcodes} does not match '
                f'manifest heldout threshold {threshold}'
            )
        manifest_train_min = manifest.get('policy', {}).get('train_min_barcodes')
        if (
            manifest_train_min is not None
            and int(manifest_train_min) != int(self.train_min_barcodes)
        ):
            raise ValueError(
                f'train_min_barcodes={self.train_min_barcodes} does not match '
                f'manifest policy {manifest_train_min}'
            )
        manifest['_resolved_expected_rows'] = expected_rows
        manifest['_resolved_heldout_threshold'] = threshold
        return manifest

    def _validate_manifest_assignments(self, df, manifest):
        id_column = self.split_id_column
        if id_column not in df.columns:
            raise ValueError(
                f'{self.datafile_path} is missing split_id_column={id_column!r}'
            )
        dataset_ids = df[id_column].astype('string').str.strip()
        if dataset_ids.isna().any() or dataset_ids.eq('').any():
            raise ValueError(f'Dataset contains blank {id_column} values')
        if dataset_ids.duplicated().any():
            raise ValueError(
                f'Dataset contains {int(dataset_ids.duplicated().sum())} duplicate '
                f'{id_column} values'
            )
        df[id_column] = dataset_ids.astype(str)
        stable_id_source_column = manifest['dataset'].get('stable_id_source_column')
        if stable_id_source_column:
            if manifest['dataset'].get('stable_id_algorithm') != 'sha256_utf8(parts_concatenated)':
                raise ValueError(
                    f"Unsupported stable-ID derivation "
                    f"{manifest['dataset'].get('stable_id_algorithm')!r}"
                )
            if stable_id_source_column not in df.columns:
                raise ValueError(
                    f'Dataset is missing stable-ID audit column '
                    f'{stable_id_source_column!r}'
                )
            source_ids = df[stable_id_source_column].astype('string').str.strip()
            if source_ids.isna().any() or source_ids.eq('').any():
                raise ValueError(
                    f'Dataset contains blank {stable_id_source_column} values'
                )
            derived_ids = source_ids.map(
                lambda value: hashlib.sha256(str(value).encode('utf-8')).hexdigest()
            )
            if not derived_ids.eq(df[id_column]).all():
                raise ValueError(
                    f'{id_column} does not match SHA256-derived '
                    f'{stable_id_source_column}'
                )
        if len(df) != int(manifest['_resolved_expected_rows']):
            raise ValueError(
                f'Dataset row count {len(df)} does not match manifest row count '
                f"{manifest['_resolved_expected_rows']}"
            )

        assignments = manifest.get('assignments')
        if not isinstance(assignments, list) or not assignments:
            raise ValueError('Split manifest assignments must be a non-empty list')
        required = {'construct_id', 'sequence', 'n_barcodes', 'partition', 'development_fold'}
        for index, assignment in enumerate(assignments):
            missing = required - set(assignment)
            if missing:
                raise ValueError(
                    f'Split assignment {index} is missing fields {sorted(missing)}'
                )
        assignment_ids = [str(row['construct_id']).strip() for row in assignments]
        if any(not stable_id for stable_id in assignment_ids):
            raise ValueError('Split manifest contains a blank construct_id')
        if len(set(assignment_ids)) != len(assignment_ids):
            raise ValueError('Split manifest contains duplicate construct_id assignments')
        dataset_id_set = set(dataset_ids.astype(str))
        assignment_id_set = set(assignment_ids)
        if dataset_id_set != assignment_id_set:
            raise ValueError(
                'Split manifest stable-ID coverage mismatch: '
                f'missing={len(dataset_id_set - assignment_id_set)}, '
                f'extra={len(assignment_id_set - dataset_id_set)}'
            )

        n_folds = int(manifest['n_development_folds'])
        allowed_partitions = {'train_only', 'development', 'audit_test'}
        assignment_by_id = {}
        for assignment, stable_id in zip(assignments, assignment_ids):
            partition = assignment['partition']
            fold = assignment['development_fold']
            if partition not in allowed_partitions:
                raise ValueError(
                    f'Unknown split partition {partition!r} for {stable_id}'
                )
            if partition == 'development':
                if fold is None or int(fold) < 0 or int(fold) >= n_folds:
                    raise ValueError(
                        f'Invalid development_fold={fold!r} for {stable_id}'
                    )
                assignment['development_fold'] = int(fold)
            elif fold is not None:
                raise ValueError(
                    f'{partition} assignment {stable_id} must have development_fold=null'
                )
            assignment_by_id[stable_id] = assignment

        sequence_column = manifest['dataset'].get('sequence_column')
        barcode_column = manifest['dataset'].get('barcode_column')
        if sequence_column not in df.columns or barcode_column not in df.columns:
            raise ValueError(
                f'Dataset is missing manifest audit fields: sequence={sequence_column!r}, '
                f'barcode={barcode_column!r}'
            )
        threshold = int(manifest['_resolved_heldout_threshold'])
        reference_generation = manifest.get('policy', {}).get(
            'assignment_reference_data_generation_id',
            manifest.get('data_generation_id'),
        )
        enforce_quality_roles = (
            reference_generation == manifest.get('data_generation_id')
        )
        for _, row in df.iterrows():
            stable_id = str(row[id_column])
            assignment = assignment_by_id[stable_id]
            observed_sequence = str(row[sequence_column]).strip().upper()
            if observed_sequence != str(assignment['sequence']).strip().upper():
                raise ValueError(f'Sequence audit mismatch for stable ID {stable_id}')
            observed_barcode = int(row[barcode_column])
            if observed_barcode != int(assignment['n_barcodes']):
                raise ValueError(f'Barcode-count audit mismatch for stable ID {stable_id}')
            if enforce_quality_roles:
                is_hq = observed_barcode >= threshold
                if assignment['partition'] == 'train_only' and is_hq:
                    raise ValueError(f'HQ stable ID {stable_id} is assigned train_only')
                if assignment['partition'] != 'train_only' and not is_hq:
                    raise ValueError(
                        f'Low-barcode stable ID {stable_id} is assigned '
                        f"{assignment['partition']}"
                    )

        expected = manifest.get('expected', {})
        if expected.get('id_hash_algorithm') != LIB1_SPLIT_ID_HASH_ALGORITHM:
            raise ValueError(
                f"Unsupported stable-ID hash algorithm {expected.get('id_hash_algorithm')!r}"
            )
        observed_all_hash = _stable_id_hash(dataset_id_set)
        if observed_all_hash != expected.get('all_ids_sha256'):
            raise ValueError('Split manifest all_ids_sha256 validation failed')
        observed_assignment_hash = _split_assignment_hash(assignments)
        if observed_assignment_hash != expected.get('assignment_sha256'):
            raise ValueError('Split manifest assignment_sha256 validation failed')

        observed_counts = {
            'total': len(assignments),
            'train_only': sum(row['partition'] == 'train_only' for row in assignments),
            'development': sum(row['partition'] == 'development' for row in assignments),
            'audit_test': sum(row['partition'] == 'audit_test' for row in assignments),
        }
        expected_counts = manifest.get('expected_counts', expected.get('counts', {}))
        for key, observed_count in observed_counts.items():
            if int(expected_counts.get(key, -1)) != int(observed_count):
                raise ValueError(
                    f'Split manifest expected count {key}={expected_counts.get(key)} '
                    f'does not match {observed_count}'
                )
        high_count = observed_counts['development'] + observed_counts['audit_test']
        if int(expected_counts.get('high_barcode', -1)) != high_count:
            raise ValueError('Split manifest high_barcode expected count is inconsistent')

        audit_ids = [
            row['construct_id']
            for row in assignments
            if row['partition'] == 'audit_test'
        ]
        if _stable_id_hash(audit_ids) != expected.get('audit_ids_sha256'):
            raise ValueError('Split manifest audit_ids_sha256 validation failed')
        development_ids = [
            row['construct_id']
            for row in assignments
            if row['partition'] == 'development'
        ]
        if _stable_id_hash(development_ids) != expected.get('development_ids_sha256'):
            raise ValueError('Split manifest development_ids_sha256 validation failed')
        train_only_ids = [
            row['construct_id']
            for row in assignments
            if row['partition'] == 'train_only'
        ]
        if _stable_id_hash(train_only_ids) != expected.get('train_only_ids_sha256'):
            raise ValueError('Split manifest train_only_ids_sha256 validation failed')

        for fold in range(n_folds):
            fold_key = str(fold)
            val_ids = [
                row['construct_id']
                for row in assignments
                if row['partition'] == 'development'
                and row['development_fold'] == fold
            ]
            train_ids = [
                row['construct_id']
                for row in assignments
                if row['partition'] == 'train_only'
                or (
                    row['partition'] == 'development'
                    and row['development_fold'] != fold
                )
            ]
            fold_expected = manifest.get('folds', {}).get(fold_key)
            if fold_expected is None:
                raise ValueError(f'Split manifest is missing fold {fold_key}')
            checks = {
                'train_pool_count': len(train_ids),
                'validation_count': len(val_ids),
                'train_pool_ids_sha256': _stable_id_hash(train_ids),
                'validation_ids_sha256': _stable_id_hash(val_ids),
            }
            for key, observed in checks.items():
                if fold_expected.get(key) != observed:
                    raise ValueError(
                        f'Split manifest fold {fold} {key} mismatch: expected '
                        f"{fold_expected.get(key)!r}, observed {observed!r}"
                    )
        return assignment_by_id

    def _validate_manifest_non_audit_assignments(self, df, manifest):
        """Validate a physically audit-excluded final-refit frame."""
        id_column = self.split_id_column
        if id_column not in df.columns:
            raise ValueError(
                f'{self.datafile_path} is missing split_id_column={id_column!r}'
            )
        dataset_ids = df[id_column].astype('string').str.strip()
        if dataset_ids.isna().any() or dataset_ids.eq('').any():
            raise ValueError(f'Dataset contains blank {id_column} values')
        if dataset_ids.duplicated().any():
            raise ValueError(f'Dataset contains duplicate {id_column} values')
        df[id_column] = dataset_ids.astype(str)

        assignments = manifest.get('assignments')
        if not isinstance(assignments, list) or not assignments:
            raise ValueError('Split manifest assignments must be a non-empty list')
        required = {
            'construct_id', 'sequence', 'n_barcodes',
            'partition', 'development_fold',
        }
        assignment_by_id = {}
        n_folds = int(manifest['n_development_folds'])
        for index, assignment in enumerate(assignments):
            missing = required - set(assignment)
            if missing:
                raise ValueError(
                    f'Split assignment {index} is missing fields {sorted(missing)}'
                )
            stable_id = str(assignment['construct_id']).strip()
            if not stable_id or stable_id in assignment_by_id:
                raise ValueError('Split manifest contains blank/duplicate construct IDs')
            partition = assignment['partition']
            fold = assignment['development_fold']
            if partition not in {'train_only', 'development', 'audit_test'}:
                raise ValueError(f'Unknown split partition {partition!r}')
            if partition == 'development':
                if fold is None or int(fold) < 0 or int(fold) >= n_folds:
                    raise ValueError(f'Invalid development fold for {stable_id}')
                assignment['development_fold'] = int(fold)
            elif fold is not None:
                raise ValueError(
                    f'{partition} assignment {stable_id} must have null fold'
                )
            assignment_by_id[stable_id] = assignment

        expected = manifest.get('expected', {})
        if expected.get('id_hash_algorithm') != LIB1_SPLIT_ID_HASH_ALGORITHM:
            raise ValueError('Unsupported stable-ID hash algorithm')
        if _split_assignment_hash(assignments) != expected.get('assignment_sha256'):
            raise ValueError('Split manifest assignment hash validation failed')
        expected_non_audit = {
            stable_id
            for stable_id, assignment in assignment_by_id.items()
            if assignment['partition'] != 'audit_test'
        }
        observed_ids = set(df[id_column].astype(str))
        if observed_ids != expected_non_audit:
            raise ValueError(
                'Physically audit-excluded frame has the wrong stable-ID coverage'
            )
        expected_count = (
            int(manifest['expected_counts']['train_only'])
            + int(manifest['expected_counts']['development'])
        )
        if len(df) != expected_count:
            raise ValueError('Physically audit-excluded frame has the wrong row count')

        stable_id_source_column = manifest['dataset'].get(
            'stable_id_source_column'
        )
        if stable_id_source_column:
            if stable_id_source_column not in df.columns:
                raise ValueError(
                    f'Dataset is missing stable-ID source {stable_id_source_column!r}'
                )
            source_ids = df[stable_id_source_column].astype('string').str.strip()
            derived_ids = source_ids.map(
                lambda value: hashlib.sha256(str(value).encode('utf-8')).hexdigest()
            )
            if not derived_ids.eq(df[id_column]).all():
                raise ValueError('Stable-ID source derivation failed')

        sequence_column = manifest['dataset']['sequence_column']
        barcode_column = manifest['dataset']['barcode_column']
        threshold = int(manifest['_resolved_heldout_threshold'])
        for _, row in df.iterrows():
            stable_id = str(row[id_column])
            assignment = assignment_by_id[stable_id]
            if str(row[sequence_column]).strip().upper() != str(
                assignment['sequence']
            ).strip().upper():
                raise ValueError(f'Sequence mismatch for non-audit ID {stable_id}')
            barcode = int(row[barcode_column])
            if barcode != int(assignment['n_barcodes']):
                raise ValueError(f'Barcode mismatch for non-audit ID {stable_id}')
            if assignment['partition'] == 'train_only' and barcode >= threshold:
                raise ValueError(f'HQ stable ID {stable_id} is assigned train_only')
            if assignment['partition'] == 'development' and barcode < threshold:
                raise ValueError(f'Low-barcode stable ID {stable_id} is development')

        audit_ids = [
            row['construct_id']
            for row in assignments
            if row['partition'] == 'audit_test'
        ]
        if _stable_id_hash(audit_ids) != expected.get('audit_ids_sha256'):
            raise ValueError('Split manifest audit-ID hash validation failed')
        return {
            stable_id: assignment_by_id[stable_id]
            for stable_id in expected_non_audit
        }

    def _split_df_manifest(self, df, manifest):
        assignment_by_id = self._validate_manifest_assignments(df, manifest)
        id_column = self.split_id_column
        partitions = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['partition']
        )
        folds = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['development_fold']
        )
        df = df.copy()
        df['_split_partition'] = partitions
        df['_development_fold'] = folds

        df_audit = df.loc[df['_split_partition'].eq('audit_test')].copy().reset_index(drop=True)
        df_val = df.loc[
            df['_split_partition'].eq('development')
            & df['_development_fold'].eq(self.split_fold)
        ].copy().reset_index(drop=True)
        df_rest = df.loc[
            df['_split_partition'].eq('train_only')
            | (
                df['_split_partition'].eq('development')
                & ~df['_development_fold'].eq(self.split_fold)
            )
        ].copy().reset_index(drop=True)
        if df_val.empty:
            raise ValueError(f'Development fold {self.split_fold} has no validation rows')
        if set(df_rest[id_column]) & set(df_val[id_column]):
            raise ValueError('Stable-ID leakage between manifest train and validation')
        if set(df_audit[id_column]) & (set(df_rest[id_column]) | set(df_val[id_column])):
            raise ValueError('Audit stable IDs leaked into train/validation')

        fold_expected = manifest['folds'][str(self.split_fold)]
        if len(df_rest) != int(fold_expected['train_pool_count']):
            raise ValueError('Manifest train-pool count differs after dataset selection')
        if _stable_id_hash(df_rest[id_column]) != fold_expected['train_pool_ids_sha256']:
            raise ValueError('Manifest train-pool stable-ID hash differs after selection')
        if len(df_val) != int(fold_expected['validation_count']):
            raise ValueError('Manifest validation count differs after dataset selection')
        if _stable_id_hash(df_val[id_column]) != fold_expected['validation_ids_sha256']:
            raise ValueError('Manifest validation stable-ID hash differs after selection')

        eligible, leftover_hq, lower_quality = self._build_train_pool_components(df_rest)
        df_train = self._subsample_train_pool(eligible, leftover_hq, lower_quality)
        self.df_train_pool = eligible.copy()
        self.df_train_pool_leftover_hq = leftover_hq.copy()
        self.df_train_pool_lower_quality = lower_quality.copy()
        self.df_audit = df_audit
        train_pool_id_hash = _stable_id_hash(eligible[id_column])
        train_final_id_hash = _stable_id_hash(df_train[id_column])
        val_id_hash = _stable_id_hash(df_val[id_column])
        audit_id_hash = _stable_id_hash(df_audit[id_column])
        self.split_summary = {
            'split_mode': 'manifest',
            'manifest_id': manifest['manifest_id'],
            'split_manifest_id': manifest['manifest_id'],
            'data_generation_id': manifest.get('data_generation_id'),
            'dataset_sha256': self.datafile_sha256,
            'split_manifest_sha256': self.split_manifest_sha256,
            'development_fold': int(self.split_fold),
            'split_id_column': id_column,
            'n_total': int(len(df)),
            'n_hq_total': int(manifest['expected_counts']['high_barcode']),
            'n_audit_excluded': int(len(df_audit)),
            'n_test': 0,
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
            'train_pool_barcode_histogram': self._barcode_histogram(eligible),
            'train_final_barcode_histogram': self._barcode_histogram(df_train),
            'train_pool_id_hash': train_pool_id_hash,
            'train_final_id_hash': train_final_id_hash,
            'val_id_hash': val_id_hash,
            'audit_id_hash': audit_id_hash,
            'normalization_id_hash': train_final_id_hash,
            # Names already harvested by train_wandb_log provenance.
            'train_pool_row_id_hash': train_pool_id_hash,
            'train_final_row_id_hash': train_final_id_hash,
            'train_row_id_hash': train_final_id_hash,
            'val_row_id_hash': val_id_hash,
            'audit_row_id_hash': audit_id_hash,
            'normalization_row_id_hash': train_final_id_hash,
            'selected_row_hash': train_final_id_hash,
        }
        empty_test = df.iloc[0:0].copy().reset_index(drop=True)
        return df_train, df_val, empty_test

    def _split_df_manifest_inner_oof(self, df, manifest):
        """Build the Stage 4 inner-checkpoint/outer-OOF development view.

        Frozen final-test rows have already been physically skipped by
        :meth:`_prep_df`.  For outer fold ``k``, fold ``(k + 1) % 5`` is the
        checkpoint-selection fold and the remaining three development folds,
        together with ``train_only``, form the candidate training pool.
        """
        n_folds = int(manifest['n_development_folds'])
        if n_folds != 5:
            raise ValueError(
                'development_inner_oof requires exactly five development folds'
            )
        assignment_by_id = self._validate_manifest_non_audit_assignments(
            df, manifest
        )
        id_column = self.split_id_column
        outer_fold = int(self.split_fold)
        inner_fold = (outer_fold + 1) % n_folds

        partitions = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['partition']
        )
        folds = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['development_fold']
        )
        df = df.copy()
        df['_split_partition'] = partitions
        df['_development_fold'] = folds

        df_oof = df.loc[
            df['_split_partition'].eq('development')
            & df['_development_fold'].eq(outer_fold)
        ].copy().reset_index(drop=True)
        df_val = df.loc[
            df['_split_partition'].eq('development')
            & df['_development_fold'].eq(inner_fold)
        ].copy().reset_index(drop=True)
        df_rest = df.loc[
            df['_split_partition'].eq('train_only')
            | (
                df['_split_partition'].eq('development')
                & ~df['_development_fold'].isin({outer_fold, inner_fold})
            )
        ].copy().reset_index(drop=True)
        if df_oof.empty or df_val.empty:
            raise ValueError(
                f'Outer fold {outer_fold} or inner fold {inner_fold} is empty'
            )

        split_sets = {
            'train': set(df_rest[id_column].astype(str)),
            'val': set(df_val[id_column].astype(str)),
            'oof': set(df_oof[id_column].astype(str)),
        }
        if (
            split_sets['train'] & split_sets['val']
            or split_sets['train'] & split_sets['oof']
            or split_sets['val'] & split_sets['oof']
        ):
            raise ValueError('Stable-ID leakage across train, inner-val, and outer-OOF')
        if set(df[id_column].astype(str)) != set().union(*split_sets.values()):
            raise ValueError('Non-final-test stable IDs are not fully partitioned')

        for label, fold, frame in (
            ('outer OOF', outer_fold, df_oof),
            ('inner validation', inner_fold, df_val),
        ):
            expected = manifest['folds'][str(fold)]
            if len(frame) != int(expected['validation_count']):
                raise ValueError(f'Manifest {label} count differs after selection')
            if _stable_id_hash(frame[id_column]) != expected['validation_ids_sha256']:
                raise ValueError(f'Manifest {label} stable-ID hash differs after selection')

        expected_train_source_ids = [
            str(row['construct_id'])
            for row in manifest['assignments']
            if row['partition'] == 'train_only'
            or (
                row['partition'] == 'development'
                and int(row['development_fold']) not in {outer_fold, inner_fold}
            )
        ]
        if len(df_rest) != len(expected_train_source_ids):
            raise ValueError('Stage 4 train-source count differs from the manifest')
        if _stable_id_hash(df_rest[id_column]) != _stable_id_hash(expected_train_source_ids):
            raise ValueError('Stage 4 train-source stable-ID hash differs from the manifest')

        eligible, _, _ = self._build_train_pool_components(df_rest)
        # The sort is part of the Stage 4 sampling contract: a given seed acts
        # on the same stable-ID order regardless of source-file row order.
        eligible = eligible.sort_values(id_column, kind='mergesort').reset_index(drop=True)
        leftover_hq = eligible.loc[
            eligible[self.barcode_column] >= self.test_min_barcodes
        ].copy().reset_index(drop=True)
        lower_quality = eligible.loc[
            eligible[self.barcode_column] < self.test_min_barcodes
        ].copy().reset_index(drop=True)
        df_train = self._subsample_train_pool(
            eligible, leftover_hq, lower_quality
        )

        self.df_train_pool = eligible.copy()
        self.df_train_pool_leftover_hq = leftover_hq.copy()
        self.df_train_pool_lower_quality = lower_quality.copy()
        self.df_audit = None
        train_source_id_hash = _stable_id_hash(df_rest[id_column])
        train_pool_id_hash = _stable_id_hash(eligible[id_column])
        train_final_id_hash = _stable_id_hash(df_train[id_column])
        val_id_hash = _stable_id_hash(df_val[id_column])
        oof_id_hash = _stable_id_hash(df_oof[id_column])
        audit_id_hash = manifest.get('expected', {}).get('audit_ids_sha256')
        audit_count = int(manifest['expected_counts']['audit_test'])
        self.split_summary = {
            'split_mode': 'manifest_development_inner_oof',
            'manifest_mode': 'development_inner_oof',
            'manifest_id': manifest['manifest_id'],
            'split_manifest_id': manifest['manifest_id'],
            'data_generation_id': manifest.get('data_generation_id'),
            'dataset_sha256': self.datafile_sha256,
            'split_manifest_sha256': self.split_manifest_sha256,
            'development_fold': outer_fold,
            'outer_development_fold': outer_fold,
            'inner_development_fold': inner_fold,
            'split_id_column': id_column,
            'n_total': int(manifest['expected_counts']['total']),
            'n_source_rows_loaded': int(len(df)),
            'n_hq_total': int(manifest['expected_counts']['high_barcode']),
            'n_audit_excluded': audit_count,
            'n_test': 0,
            'n_oof': int(len(df_oof)),
            'n_val': int(len(df_val)),
            'n_train_source': int(len(df_rest)),
            'n_train_pool_eligible': int(len(eligible)),
            'n_train_pool_leftover_hq': int(len(leftover_hq)),
            'n_train_pool_lower_quality': int(len(lower_quality)),
            'n_train_final': int(len(df_train)),
            'n_train_final_hq': int(
                (df_train[self.barcode_column] >= self.test_min_barcodes).sum()
            ),
            'n_train_final_lower_quality': int(
                (df_train[self.barcode_column] < self.test_min_barcodes).sum()
            ),
            'train_sampling_mode': self.train_sampling_mode,
            'train_subsample_seed': int(self.train_subsample_seed),
            'train_size_frac': float(self.train_size_frac),
            'train_size_n': (
                None if self.train_size_n is None else int(self.train_size_n)
            ),
            'train_min_barcodes': int(self.train_min_barcodes),
            'train_max_barcodes': (
                None
                if self.train_max_barcodes is None
                else int(self.train_max_barcodes)
            ),
            'test_min_barcodes': int(self.test_min_barcodes),
            'train_pool_barcode_histogram': self._barcode_histogram(eligible),
            'train_final_barcode_histogram': self._barcode_histogram(df_train),
            'stable_sampling_order': f'{id_column}_ascending',
            'nested_prefix_sampling': self.train_sampling_mode == 'random',
            'train_source_id_hash': train_source_id_hash,
            'train_pool_id_hash': train_pool_id_hash,
            'train_final_id_hash': train_final_id_hash,
            'val_id_hash': val_id_hash,
            'oof_id_hash': oof_id_hash,
            'audit_id_hash': audit_id_hash,
            'normalization_id_hash': train_final_id_hash,
            'train_source_row_id_hash': train_source_id_hash,
            'train_pool_row_id_hash': train_pool_id_hash,
            'train_final_row_id_hash': train_final_id_hash,
            'train_row_id_hash': train_final_id_hash,
            'val_row_id_hash': val_id_hash,
            'oof_row_id_hash': oof_id_hash,
            'audit_row_id_hash': audit_id_hash,
            'normalization_row_id_hash': train_final_id_hash,
            'selected_row_hash': train_final_id_hash,
            'final_test_rows_physically_excluded': True,
            'audit_loader_authorized': False,
        }
        return df_train, df_val, df_oof

    def _split_df_manifest_final(self, df, manifest):
        """Build the locked all-development refit or one-time audit view."""
        if self.manifest_mode == 'final_refit':
            assignment_by_id = self._validate_manifest_non_audit_assignments(
                df, manifest
            )
        else:
            assignment_by_id = self._validate_manifest_assignments(df, manifest)
        id_column = self.split_id_column
        partitions = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['partition']
        )
        folds = df[id_column].map(
            lambda stable_id: assignment_by_id[str(stable_id)]['development_fold']
        )
        df = df.copy()
        df['_split_partition'] = partitions
        df['_development_fold'] = folds

        df_audit = df.loc[
            df['_split_partition'].eq('audit_test')
        ].copy().reset_index(drop=True)
        df_non_audit = df.loc[
            ~df['_split_partition'].eq('audit_test')
        ].copy().reset_index(drop=True)
        if set(df_audit[id_column]) & set(df_non_audit[id_column]):
            raise ValueError('Audit stable IDs leaked into the final-refit pool')

        eligible, leftover_hq, lower_quality = self._build_train_pool_components(
            df_non_audit
        )
        df_train = self._subsample_train_pool(
            eligible, leftover_hq, lower_quality
        )
        if len(df_train) != len(eligible):
            raise ValueError(
                'final_refit/audit_eval requires train_size_frac=1 and no '
                'train_size_n subsampling'
            )

        self.df_train_pool = eligible.copy()
        self.df_train_pool_leftover_hq = leftover_hq.copy()
        self.df_train_pool_lower_quality = lower_quality.copy()
        self.df_audit = df_audit.copy() if self.manifest_mode == 'audit_eval' else None
        train_pool_id_hash = _stable_id_hash(eligible[id_column])
        train_final_id_hash = _stable_id_hash(df_train[id_column])
        expected_audit_hash = manifest.get('expected', {}).get('audit_ids_sha256')
        audit_id_hash = (
            _stable_id_hash(df_audit[id_column])
            if self.manifest_mode == 'audit_eval'
            else expected_audit_hash
        )
        if audit_id_hash != expected_audit_hash:
            raise ValueError('Final-refit audit exclusion hash mismatch')

        audit_authorized = self.manifest_mode == 'audit_eval'
        empty = df.iloc[0:0].copy().reset_index(drop=True)
        df_test = df_audit.copy() if audit_authorized else empty.copy()
        audit_count = int(manifest['expected_counts']['audit_test'])
        self.split_summary = {
            'split_mode': f'manifest_{self.manifest_mode}',
            'manifest_id': manifest['manifest_id'],
            'split_manifest_id': manifest['manifest_id'],
            'data_generation_id': manifest.get('data_generation_id'),
            'dataset_sha256': self.datafile_sha256,
            'split_manifest_sha256': self.split_manifest_sha256,
            'development_fold': None,
            'split_id_column': id_column,
            'n_total': int(manifest['expected_counts']['total']),
            'n_source_rows_loaded': int(len(df)),
            'n_hq_total': int(manifest['expected_counts']['high_barcode']),
            'n_audit_excluded': audit_count,
            'n_test': int(len(df_test)),
            'n_val': 0,
            'n_train_pool_eligible': int(len(eligible)),
            'n_train_pool_leftover_hq': int(len(leftover_hq)),
            'n_train_pool_lower_quality': int(len(lower_quality)),
            'n_train_final': int(len(df_train)),
            'n_train_final_hq': int(
                (df_train[self.barcode_column] >= self.test_min_barcodes).sum()
            ),
            'n_train_final_lower_quality': int(
                (df_train[self.barcode_column] < self.test_min_barcodes).sum()
            ),
            'train_sampling_mode': self.train_sampling_mode,
            'train_subsample_seed': int(self.train_subsample_seed),
            'train_size_frac': float(self.train_size_frac),
            'train_size_n': (
                None if self.train_size_n is None else int(self.train_size_n)
            ),
            'train_min_barcodes': int(self.train_min_barcodes),
            'train_max_barcodes': (
                None
                if self.train_max_barcodes is None
                else int(self.train_max_barcodes)
            ),
            'test_min_barcodes': int(self.test_min_barcodes),
            'train_pool_barcode_histogram': self._barcode_histogram(eligible),
            'train_final_barcode_histogram': self._barcode_histogram(df_train),
            'train_pool_id_hash': train_pool_id_hash,
            'train_final_id_hash': train_final_id_hash,
            'audit_id_hash': audit_id_hash,
            'train_pool_row_id_hash': train_pool_id_hash,
            'train_final_row_id_hash': train_final_id_hash,
            'train_row_id_hash': train_final_id_hash,
            'val_row_id_hash': '',
            'audit_row_id_hash': audit_id_hash,
            'normalization_row_id_hash': train_final_id_hash,
            'selected_row_hash': train_final_id_hash,
            'audit_loader_authorized': audit_authorized,
        }
        return df_train, empty, df_test

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
        self._validate_datafile_hash()
        self.split_manifest = self._load_split_manifest()
        df = self._prep_df()
        df_oof = None
        if self.split_manifest is None:
            df_train, df_val, df_test = self._split_df(df)
            self.split_summary['split_mode'] = 'legacy_random'
            self.split_summary['dataset_sha256'] = self.datafile_sha256
            self.split_summary['split_manifest_sha256'] = None
        else:
            if self.manifest_mode == 'development':
                df_train, df_val, df_test = self._split_df_manifest(
                    df, self.split_manifest
                )
            elif self.manifest_mode == 'development_inner_oof':
                df_train, df_val, df_oof = self._split_df_manifest_inner_oof(
                    df, self.split_manifest
                )
                # Deliberately do not create even an empty test frame in this
                # mode.  The final-test partition was never parsed.
                df_test = None
            else:
                df_train, df_val, df_test = self._split_df_manifest_final(
                    df, self.split_manifest
                )
        if df_oof is None:
            df_train, df_val, df_test = self._standardize_targets(
                df_train, df_val, df_test
            )
        else:
            df_train, df_val, df_oof = self._standardize_targets(
                df_train, df_val, df_oof
            )
        if isinstance(self.split_summary, dict):
            self.split_summary['target_normalization_enabled'] = bool(self.normalize)
            self.split_summary['target_normalization_mean'] = (
                {key: float(value) for key, value in self.target_mean.items()}
                if isinstance(self.target_mean, dict)
                else float(self.target_mean)
            )
            self.split_summary['target_normalization_std'] = (
                {key: float(value) for key, value in self.target_std.items()}
                if isinstance(self.target_std, dict)
                else float(self.target_std)
            )
            self.split_summary['target_normalization_std_ddof'] = 1
            self.split_summary['target_normalization_row_count'] = int(len(df_train))

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.df_oof = df_oof

        self.dataset_train = self._df_to_dataset(df_train, training=True)
        self.dataset_train_eval = self._df_to_dataset(df_train, training=False)
        self.dataset_val = (
            None if df_val.empty else self._df_to_dataset(df_val, training=False)
        )
        self.dataset_test = (
            None
            if df_test is None or df_test.empty
            else self._df_to_dataset(df_test, training=False)
        )
        self.dataset_oof = (
            None
            if df_oof is None or df_oof.empty
            else self._df_to_dataset(df_oof, training=False)
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def train_eval_dataloader(self):
        dataset = self.dataset_train_eval if self.dataset_train_eval is not None else self.dataset_train
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if self.dataset_val is None:
            return None
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        if self.dataset_test is None:
            return None
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def oof_dataloader(self):
        if self.dataset_oof is None:
            return None
        return DataLoader(self.dataset_oof, batch_size=self.batch_size, shuffle=False,
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
