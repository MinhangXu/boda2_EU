import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from boda.common import utils
from boda.data import BashorDataModule
from src.learn.generate_lib1_dedup_split_manifests import (
    build_split_manifest,
    sha256_file,
)
from src.learn.prepare_lib1_single_part_datasets import (
    PART_SPECS,
    TARGET_COLUMN,
    prepare_frame,
    stable_construct_id,
)


class Lib1DedupPreparationTests(unittest.TestCase):
    def test_shared_prep_uses_aggregate_count_formula_and_filters_invalid_rows(self):
        spec = replace(
            PART_SPECS["intron"],
            part="Synthetic",
            selected_length=None,
            length_policy="all",
        )
        raw = pd.DataFrame(
            {
                "parts_concatenated": ["construct-a", "construct-b", "bad-x", "zero-dna"],
                "Intron": ["ACGT", "TGCA", "x", "AAAA"],
                "number_of_barcodes": [1, 8, 20, 3],
                "DNA_bc_counts_sum": [2, 8, 4, 0],
                "RNA_bc_counts_sum": [8, 2, 10, 4],
                # Deliberately wrong: canonical preparation must recompute it.
                "RNA/DNA": [999.0, 999.0, 999.0, 999.0],
            }
        )
        prepared, diagnostics = prepare_frame(raw, spec)

        self.assertEqual(len(prepared), 2)
        self.assertEqual(diagnostics["high_barcode_rows"], 1)
        np.testing.assert_allclose(prepared["RNA_DNA"], [4.0, 0.25])
        np.testing.assert_allclose(prepared[TARGET_COLUMN], [2.0, -2.0])
        self.assertEqual(
            prepared.loc[0, "construct_id"], stable_construct_id("construct-a")
        )

    @unittest.skipUnless(
        PART_SPECS["enhancer"].dedup_source_path.exists(),
        "workspace-external Lib1 dedup tables are unavailable",
    )
    def test_canonical_five_part_counts_and_target_formula(self):
        for part_slug, spec in PART_SPECS.items():
            with self.subTest(part=part_slug):
                raw = pd.read_csv(spec.dedup_source_path)
                prepared, diagnostics = prepare_frame(raw, spec)
                self.assertEqual(len(prepared), spec.expected_row_count)
                self.assertEqual(
                    diagnostics["high_barcode_rows"],
                    spec.expected_high_barcode_count,
                )
                expected = np.log2(
                    prepared["RNA_bc_counts_sum"].to_numpy(dtype=float)
                    / prepared["DNA_bc_counts_sum"].to_numpy(dtype=float)
                )
                np.testing.assert_allclose(
                    prepared[TARGET_COLUMN].to_numpy(), expected, rtol=0, atol=1e-12
                )
                self.assertFalse(prepared[spec.sequence_column].eq("X").any())


class Lib1FrozenSplitTests(unittest.TestCase):
    @staticmethod
    def synthetic_frame():
        n_rows = 14
        return pd.DataFrame(
            {
                "construct_id": [stable_construct_id(f"construct-{i}") for i in range(n_rows)],
                "parts_concatenated": [f"construct-{i}" for i in range(n_rows)],
                "Sequence": ["ACGT" if i % 2 == 0 else "TGCA" for i in range(n_rows)],
                "n_barcodes": [1] * 6 + [8] * 8,
                "log2_RNA_DNA": np.linspace(-2.0, 2.0, n_rows),
                "log10_RNA_DNA": np.linspace(-1.0, 1.0, n_rows),
            }
        )

    def build_manifest(self, frame, data_path, dataset_sha256, n_folds=2):
        return build_split_manifest(
            frame,
            part="Synthetic",
            part_slug="synthetic",
            dataset_path=data_path,
            dataset_sha256=dataset_sha256,
            data_generation_id="lib1_single_part_dedup_exact_v1",
            id_column="construct_id",
            sequence_column="Sequence",
            heldout_min_barcodes=8,
            train_min_barcodes=1,
            padded_seq_len=4,
            padding_mode="none",
            neutral_pad_char="N",
            normalize=True,
            split_seed=20260709,
            n_folds=n_folds,
            audit_size=2,
        )

    def test_split_is_row_order_independent_reproducible_and_leak_free(self):
        frame = self.synthetic_frame()
        first = self.build_manifest(frame, Path("synthetic.tsv"), "a" * 64, n_folds=3)
        second = self.build_manifest(
            frame.iloc[::-1].reset_index(drop=True),
            Path("synthetic.tsv"),
            "a" * 64,
            n_folds=3,
        )
        self.assertEqual(first, second)

        assignments = first["assignments"]
        audit_ids = {
            row["construct_id"]
            for row in assignments
            if row["partition"] == "audit_test"
        }
        all_val_ids = []
        for fold in range(3):
            val_ids = {
                row["construct_id"]
                for row in assignments
                if row["partition"] == "development"
                and row["development_fold"] == fold
            }
            train_ids = {
                row["construct_id"]
                for row in assignments
                if row["partition"] == "train_only"
                or (
                    row["partition"] == "development"
                    and row["development_fold"] != fold
                )
            }
            self.assertFalse(train_ids & val_ids)
            self.assertFalse(audit_ids & (train_ids | val_ids))
            all_val_ids.extend(val_ids)
        development_ids = [
            row["construct_id"]
            for row in assignments
            if row["partition"] == "development"
        ]
        self.assertCountEqual(all_val_ids, development_ids)

    def test_manifest_datamodule_excludes_audit_has_no_test_loader_and_rc_is_train_only(self):
        frame = self.synthetic_frame()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "synthetic.tsv"
            frame.to_csv(data_path, sep="\t", index=False)
            dataset_sha256 = sha256_file(data_path)
            manifest = self.build_manifest(frame, data_path, dataset_sha256)
            manifest_path = tmp_path / "split.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            split_sha256 = sha256_file(manifest_path)

            data = BashorDataModule(
                datafile_path=str(data_path),
                sep="\t",
                sequence_column="Sequence",
                target_column="log2_RNA_DNA",
                barcode_column="n_barcodes",
                padded_seq_len=4,
                padding_mode="none",
                batch_size=4,
                num_workers=0,
                normalize=True,
                split_manifest_path=str(manifest_path),
                split_fold=0,
                split_id_column="construct_id",
                expected_data_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                test_min_barcodes=8,
                train_min_barcodes=1,
                min_train_size=1,
                use_reverse_complements=True,
            )
            data.setup()

            train_ids = set(data.df_train["construct_id"])
            val_ids = set(data.df_val["construct_id"])
            audit_ids = set(data.df_audit["construct_id"])
            self.assertFalse(train_ids & val_ids)
            self.assertFalse(audit_ids & (train_ids | val_ids))
            self.assertIsNone(data.dataset_test)
            self.assertIsNone(data.test_dataloader())
            self.assertEqual(len(data.dataset_train), 2 * len(data.df_train))
            self.assertEqual(len(data.dataset_train_eval), len(data.df_train))
            self.assertEqual(len(data.dataset_val), len(data.df_val))
            self.assertAlmostEqual(float(data.df_train["target_processed"].mean()), 0.0)
            self.assertEqual(
                data.split_summary["normalization_id_hash"],
                data.split_summary["train_final_id_hash"],
            )

            original_x, original_y = data.dataset_train[0]
            rc_x, rc_y = data.dataset_train[1]
            self.assertTrue(torch.equal(original_y, rc_y))
            self.assertTrue(
                torch.equal(rc_x, utils.reverse_complement_onehot(original_x))
            )

            bad_hash_data = BashorDataModule(
                datafile_path=str(data_path),
                sep="\t",
                sequence_column="Sequence",
                target_column="log2_RNA_DNA",
                barcode_column="n_barcodes",
                padded_seq_len=4,
                padding_mode="none",
                num_workers=0,
                split_manifest_path=str(manifest_path),
                expected_data_sha256="0" * 64,
                test_min_barcodes=8,
                train_min_barcodes=1,
                min_train_size=1,
            )
            with self.assertRaisesRegex(ValueError, "Dataset SHA256 mismatch"):
                bad_hash_data.setup()

            wrong_target_data = BashorDataModule(
                datafile_path=str(data_path),
                sep="\t",
                sequence_column="Sequence",
                target_column="log10_RNA_DNA",
                barcode_column="n_barcodes",
                padded_seq_len=4,
                padding_mode="none",
                num_workers=0,
                normalize=True,
                split_manifest_path=str(manifest_path),
                expected_data_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                test_min_barcodes=8,
                train_min_barcodes=1,
                min_train_size=1,
            )
            with self.assertRaisesRegex(ValueError, "target_column"):
                wrong_target_data.setup()

    def test_manifest_final_refit_and_separate_audit_eval_modes(self):
        frame = self.synthetic_frame()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_path = tmp_path / "synthetic.tsv"
            frame.to_csv(data_path, sep="\t", index=False)
            dataset_sha256 = sha256_file(data_path)
            manifest = self.build_manifest(frame, data_path, dataset_sha256)
            manifest_path = tmp_path / "split.json"
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            )
            split_sha256 = sha256_file(manifest_path)

            common = dict(
                datafile_path=str(data_path),
                sep="\t",
                sequence_column="Sequence",
                target_column="log2_RNA_DNA",
                barcode_column="n_barcodes",
                padded_seq_len=4,
                padding_mode="none",
                batch_size=4,
                num_workers=0,
                normalize=True,
                split_manifest_path=str(manifest_path),
                split_id_column="construct_id",
                expected_data_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                test_min_barcodes=8,
                train_min_barcodes=1,
                min_train_size=1,
                train_sampling_mode="random",
                train_size_frac=1.0,
            )
            final_refit = BashorDataModule(
                **common, manifest_mode="final_refit", use_reverse_complements=False
            )
            final_refit.setup()
            audit_ids = {
                row["construct_id"]
                for row in manifest["assignments"]
                if row["partition"] == "audit_test"
            }
            train_ids = set(final_refit.df_train["construct_id"])
            self.assertIsNone(final_refit.df_audit)
            self.assertFalse(audit_ids & train_ids)
            self.assertEqual(len(final_refit.df_train), len(frame) - len(audit_ids))
            self.assertEqual(len(final_refit.df_val), 0)
            self.assertEqual(len(final_refit.df_test), 0)
            self.assertIsNone(final_refit.val_dataloader())
            self.assertIsNone(final_refit.test_dataloader())
            self.assertEqual(final_refit.split_summary["n_test"], 0)
            self.assertFalse(final_refit.split_summary["audit_loader_authorized"])

            audit_eval = BashorDataModule(
                **common, manifest_mode="audit_eval", use_reverse_complements=False
            )
            audit_eval.setup()
            self.assertEqual(set(audit_eval.df_train["construct_id"]), train_ids)
            self.assertEqual(set(audit_eval.df_test["construct_id"]), audit_ids)
            self.assertEqual(audit_eval.target_mean, final_refit.target_mean)
            self.assertEqual(audit_eval.target_std, final_refit.target_std)
            self.assertIsNotNone(audit_eval.test_dataloader())
            self.assertEqual(audit_eval.split_summary["n_test"], len(audit_ids))
            self.assertTrue(audit_eval.split_summary["audit_loader_authorized"])


if __name__ == "__main__":
    unittest.main()
