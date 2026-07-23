import argparse
import json
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from boda.data import BashorDataModule
from src.learn import train_wandb_log as training
from src.learn.generate_lib1_dedup_split_manifests import (
    build_split_manifest,
    sha256_file,
)
from src.learn.prepare_lib1_single_part_datasets import stable_construct_id


class Stage4DataContractTests(unittest.TestCase):
    @staticmethod
    def synthetic_frame():
        n_rows = 32
        return pd.DataFrame(
            {
                'construct_id': [
                    stable_construct_id(f'construct-{index}')
                    for index in range(n_rows)
                ],
                'parts_concatenated': [
                    f'construct-{index}' for index in range(n_rows)
                ],
                'Sequence': [
                    'ACGTACGT' if index % 2 == 0 else 'TGCATGCA'
                    for index in range(n_rows)
                ],
                'n_barcodes': [1] * 7 + [8] * 25,
                'log2_RNA_DNA': np.linspace(-3.0, 3.0, n_rows),
            }
        )

    @staticmethod
    def write_fixture(root, frame):
        data_path = root / 'synthetic.tsv'
        frame.to_csv(data_path, sep='\t', index=False)
        dataset_sha256 = sha256_file(data_path)
        manifest = build_split_manifest(
            frame,
            part='Synthetic',
            part_slug='synthetic',
            dataset_path=data_path,
            dataset_sha256=dataset_sha256,
            data_generation_id='lib1_single_part_dedup_exact_v1',
            id_column='construct_id',
            sequence_column='Sequence',
            heldout_min_barcodes=8,
            train_min_barcodes=1,
            padded_seq_len=8,
            padding_mode='none',
            neutral_pad_char='N',
            normalize=True,
            split_seed=20260709,
            n_folds=5,
            audit_size=5,
        )
        manifest_path = root / 'split.json'
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(',', ':')) + '\n'
        )
        return data_path, manifest_path, manifest

    @staticmethod
    def data_module(data_path, manifest_path, *, size, seed=31415, fold=2):
        return BashorDataModule(
            datafile_path=str(data_path),
            sep='\t',
            sequence_column='Sequence',
            target_column='log2_RNA_DNA',
            barcode_column='n_barcodes',
            padded_seq_len=8,
            padding_mode='none',
            batch_size=4,
            num_workers=0,
            normalize=True,
            split_manifest_path=str(manifest_path),
            manifest_mode='development_inner_oof',
            split_fold=fold,
            split_id_column='construct_id',
            expected_data_sha256=sha256_file(data_path),
            expected_split_sha256=sha256_file(manifest_path),
            test_min_barcodes=8,
            train_min_barcodes=1,
            train_size_n=size,
            min_train_size=1,
            train_sampling_mode='random',
            train_subsample_seed=seed,
            use_reverse_complements=False,
        )

    def test_inner_outer_routing_normalization_and_physical_exclusion(self):
        frame = self.synthetic_frame()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path, manifest_path, manifest = self.write_fixture(root, frame)
            audit_ids = {
                row['construct_id']
                for row in manifest['assignments']
                if row['partition'] == 'audit_test'
            }
            expected_oof = {
                row['construct_id']
                for row in manifest['assignments']
                if row['partition'] == 'development'
                and row['development_fold'] == 2
            }
            expected_inner = {
                row['construct_id']
                for row in manifest['assignments']
                if row['partition'] == 'development'
                and row['development_fold'] == 3
            }

            data = self.data_module(
                data_path, manifest_path, size=8, seed=31415, fold=2
            )
            with mock.patch(
                'boda.data.bashor_datamodule.pd.read_csv', wraps=pd.read_csv
            ) as read_csv:
                data.setup()

            skiprow_reads = [
                call
                for call in read_csv.call_args_list
                if 'skiprows' in call[-1]
            ]
            self.assertEqual(len(skiprow_reads), 1)
            self.assertEqual(
                len(skiprow_reads[0][-1]['skiprows']), len(audit_ids)
            )
            self.assertEqual(set(data.df_oof['construct_id']), expected_oof)
            self.assertEqual(set(data.df_val['construct_id']), expected_inner)
            self.assertFalse(
                set(data.df_train['construct_id'])
                & (expected_oof | expected_inner | audit_ids)
            )
            self.assertIsNone(data.df_audit)
            self.assertIsNone(data.df_test)
            self.assertIsNone(data.dataset_test)
            self.assertIsNone(data.test_dataloader())
            self.assertIsNotNone(data.dataset_oof)
            self.assertIsNotNone(data.oof_dataloader())

            expected_mean = float(data.df_train['log2_RNA_DNA'].mean())
            expected_std = float(data.df_train['log2_RNA_DNA'].std())
            self.assertAlmostEqual(data.target_mean, expected_mean)
            self.assertAlmostEqual(data.target_std, expected_std)
            np.testing.assert_allclose(
                data.df_oof['target_processed'],
                (data.df_oof['log2_RNA_DNA'] - expected_mean) / expected_std,
            )
            summary = data.split_summary
            self.assertEqual(summary['outer_development_fold'], 2)
            self.assertEqual(summary['inner_development_fold'], 3)
            self.assertEqual(summary['n_source_rows_loaded'], len(frame) - len(audit_ids))
            self.assertEqual(summary['n_test'], 0)
            self.assertEqual(summary['n_oof'], len(expected_oof))
            self.assertTrue(summary['final_test_rows_physically_excluded'])
            self.assertFalse(summary['audit_loader_authorized'])
            self.assertEqual(
                summary['normalization_row_id_hash'],
                summary['train_final_row_id_hash'],
            )

    def test_random_subsets_are_nested_and_source_row_order_independent(self):
        frame = self.synthetic_frame()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_root = root / 'first'
            second_root = root / 'second'
            first_root.mkdir()
            second_root.mkdir()
            first_data, first_manifest, _ = self.write_fixture(first_root, frame)
            shuffled = frame.sample(frac=1.0, random_state=91).reset_index(drop=True)
            second_data, second_manifest, _ = self.write_fixture(second_root, shuffled)

            small = self.data_module(first_data, first_manifest, size=4, seed=22)
            large = self.data_module(first_data, first_manifest, size=12, seed=22)
            reordered = self.data_module(second_data, second_manifest, size=4, seed=22)
            small.setup()
            large.setup()
            reordered.setup()

            small_ids = set(small.df_train['construct_id'])
            large_ids = set(large.df_train['construct_id'])
            reordered_ids = set(reordered.df_train['construct_id'])
            self.assertTrue(small_ids < large_ids)
            self.assertEqual(small_ids, reordered_ids)
            self.assertEqual(
                small.split_summary['train_final_id_hash'],
                reordered.split_summary['train_final_id_hash'],
            )
            self.assertEqual(
                small.split_summary['stable_sampling_order'],
                'construct_id_ascending',
            )
            self.assertTrue(small.split_summary['nested_prefix_sampling'])


class Stage4RuntimeContractTests(unittest.TestCase):
    @staticmethod
    def valid_namespaces():
        main = argparse.Namespace(
            campaign_stage='stage4_downsampling',
            part_slug='promoter',
            logger_project=(
                'promoter__bashor_in_house__dedup_exact_v1__'
                'stage4_downsampling_development'
            ),
            evaluate_test_after_fit=False,
            prediction_splits=['oof'],
            epoch_eval_splits=['train', 'val'],
            rc_mode='off',
            training_regime='scratch',
            loss_mode='barcode_weighted_mse',
            graph_module='CNNWeightedRegressionTraining',
        )
        data = argparse.Namespace(
            manifest_mode='development_inner_oof',
            split_manifest_path='/frozen/split.json',
            train_sampling_mode='random',
            train_min_barcodes=1,
            train_max_barcodes=None,
            train_size_frac=1.0,
            use_reverse_complements=False,
            barcode_weighting=True,
            barcode_weight_cap=8.0,
            barcode_weight_min=0.1,
        )
        return main, data

    def test_valid_weighted_and_unweighted_routes(self):
        main, data = self.valid_namespaces()
        training._validate_stage4_downsampling_contract(main, data)

        main.part_slug = 'enhancer'
        main.logger_project = (
            'enhancer__bashor_in_house__dedup_exact_v1__'
            'stage4_downsampling_development'
        )
        main.training_regime = 'transfer'
        main.loss_mode = 'unweighted_mse'
        main.graph_module = 'CNNBassetBranchedScopedTransfer'
        main.rc_mode = 'on'
        data.barcode_weighting = False
        data.use_reverse_complements = True
        training._validate_stage4_downsampling_contract(main, data)

    def test_runtime_contract_rejects_leakage_and_nominal_weighting(self):
        mutations = (
            ('prediction_splits', ['val'], 'prediction_splits'),
            ('epoch_eval_splits', ['val'], 'epoch_eval_splits'),
            ('evaluate_test_after_fit', True, 'cannot evaluate'),
            ('logger_project', 'promoter__wrong', 'part/project'),
            ('rc_mode', 'on', 'disagrees'),
            ('graph_module', 'CNNBasicTraining', 'requires graph_module'),
        )
        for field, value, message in mutations:
            main, data = self.valid_namespaces()
            setattr(main, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                training._validate_stage4_downsampling_contract(main, data)

        main, data = self.valid_namespaces()
        data.manifest_mode = 'development'
        with self.assertRaisesRegex(ValueError, 'development_inner_oof'):
            training._validate_stage4_downsampling_contract(main, data)

        main, data = self.valid_namespaces()
        data.barcode_weighting = False
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            training._validate_stage4_downsampling_contract(main, data)

    def test_campaign_project_suffix_is_registered(self):
        main, _ = self.valid_namespaces()
        main.campaign_id = training.LIB1_DEDUP_CAMPAIGN_ID
        main.logger_type = 'wandb'
        main.wandb_entity = training.LIB1_DEDUP_WANDB_ENTITY
        training._validate_campaign_wandb_contract(main)

    @unittest.skipUnless(
        training.LIB1_DEDUP_STAGE4_MANIFEST_PATH.is_file(),
        'frozen Stage 4 manifest is unavailable',
    )
    def test_exact_manifest_row_argv_and_registry_binding(self):
        row = json.loads(
            training.LIB1_DEDUP_STAGE4_MANIFEST_PATH.read_text().splitlines()[0]
        )
        argv = shlex.split(row['train_command'])[1:]
        main = argparse.Namespace(
            campaign_stage='stage4_downsampling', cell_id=row['cell_id']
        )
        environment = {
            'BODA_CONFIG_PATH': str(
                training.LIB1_DEDUP_STAGE4_MANIFEST_PATH.resolve()
            ),
            'BODA_CONFIG_MANIFEST_SHA256': (
                training.LIB1_DEDUP_STAGE4_MANIFEST_SHA256
            ),
            'BODA_MANIFEST_ROW': str(row['row']),
            'BODA_MANIFEST_ROW_FINGERPRINT': row['row_fingerprint'],
            'BODA_RUNTIME_ARGV_SHA256': training._runtime_argv_sha256(argv),
            'BODA_RUNS_CSV': str(
                training.LIB1_DEDUP_STAGE4_REGISTRY_PATH.resolve()
            ),
            'BODA_LAUNCH_SCRIPT': (
                'run_lib1_dedup_stage4_downsampling_campaign.py'
            ),
        }
        with mock.patch.dict('os.environ', environment, clear=False), mock.patch.object(
            sys, 'argv', argv
        ):
            bound = training._validate_stage4_manifest_launch_contract(main)
        self.assertEqual(bound['row_fingerprint'], row['row_fingerprint'])

        bad_environments = (
            ('BODA_MANIFEST_ROW', '2', 'cell_id'),
            ('BODA_MANIFEST_ROW_FINGERPRINT', 'wrong', 'fingerprint'),
            ('BODA_RUNTIME_ARGV_SHA256', 'wrong', 'argv SHA256'),
            ('BODA_RUNS_CSV', '/tmp/global-runs.csv', 'Stage4-only'),
        )
        for field, value, message in bad_environments:
            changed = dict(environment)
            changed[field] = value
            with self.subTest(field=field), mock.patch.dict(
                'os.environ', changed, clear=False
            ), mock.patch.object(sys, 'argv', argv), self.assertRaisesRegex(
                ValueError, message
            ):
                training._validate_stage4_manifest_launch_contract(main)

    def test_stage4_provenance_columns_are_append_only(self):
        self.assertEqual(
            training.RUNS_CSV_COLUMNS[-7:],
            [
                'config_manifest_sha256',
                'manifest_row',
                'manifest_row_fingerprint',
                'runtime_argv_sha256',
                'resolved_arguments_sha256',
                'run_registry_path',
                'optimizer_steps',
            ],
        )


if __name__ == '__main__':
    unittest.main()
