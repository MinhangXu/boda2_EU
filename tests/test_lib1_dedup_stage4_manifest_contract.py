import argparse
import json
import shlex
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from src.learn.run_lib1_dedup_stage4_downsampling_campaign import (
    EXPECTED_MANIFEST_SHA256,
    REQUIRED_PILOT_CELL,
    STAGE4_RUNS_CSV,
    provenance_scalar_text,
    read_registry,
    sha256_file,
)
from src.learn.verify_lib1_dedup_stage4_downsampling_manifest import validate


REPO = Path(__file__).resolve().parents[1]
LEARN = REPO / "src/learn"
PREFIX = LEARN / "outputs/hpo_manifests/lib1_dedup_stage4_downsampling_july2026"
MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")


def verifier_args():
    return argparse.Namespace(
        manifest=MANIFEST,
        portfolio=Path(str(PREFIX) + "__portfolio.json"),
        summary=Path(str(PREFIX) + "__summary.json"),
        report=Path(str(PREFIX) + "__validation_report.json"),
    )


@unittest.skipUnless(MANIFEST.is_file(), "frozen Stage 4 manifest is unavailable")
class Stage4ManifestContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line]

    def test_independent_verifier_and_exact_hash(self):
        report = validate(verifier_args())
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["manifest_sha256"], EXPECTED_MANIFEST_SHA256)
        self.assertEqual(sha256_file(MANIFEST), EXPECTED_MANIFEST_SHA256)
        self.assertEqual(report["rows"], 660)
        self.assertEqual(report["configs"], 15)
        self.assertEqual(report["nested_prefix_tracks_checked"], 75)
        self.assertFalse(report["final_test_loader_instantiated"])
        self.assertFalse(report["final_test_metrics_read"])
        self.assertEqual(report["commands_executed"], 0)

    def test_tiered_accounting_and_six_point_primary_grid(self):
        self.assertEqual(
            Counter(row["stage4_lane"] for row in self.rows),
            {"primary": 400, "alternative": 180, "scratch_diagnostic": 80},
        )
        self.assertEqual(
            Counter(row["part_slug"] for row in self.rows),
            {"enhancer": 200, "promoter": 120, "intron": 120, "utr3": 120, "utr5": 100},
        )
        primary_sizes = {
            row["downsample_n_label"] for row in self.rows if row["stage4_lane"] == "primary"
        }
        alternative_sizes = {
            row["downsample_n_label"] for row in self.rows if row["stage4_lane"] == "alternative"
        }
        self.assertEqual(primary_sizes, {"40", "250", "400", "2500", "4000", "full"})
        self.assertEqual(alternative_sizes, {"40", "400", "4000", "full"})

        primary_configs = {
            (row["part_slug"], row["base_config_id"])
            for row in self.rows if row["stage4_lane"] == "primary"
        }
        alternatives = {
            (row["part_slug"], row["base_config_id"])
            for row in self.rows if row["stage4_lane"] == "alternative"
        }
        scratch = {
            (row["part_slug"], row["base_config_id"])
            for row in self.rows if row["stage4_lane"] == "scratch_diagnostic"
        }
        self.assertEqual(len(primary_configs), 5)
        self.assertEqual(len(alternatives), 9)
        self.assertEqual(len(scratch), 1)

    def test_inner_outer_and_sampling_contract(self):
        for row in self.rows:
            self.assertEqual(row["inner_validation_fold"], (row["outer_oof_fold"] + 1) % 5)
            self.assertEqual(row["manifest_mode"], "development_inner_oof")
            self.assertEqual(row["prediction_splits"], ["oof"])
            self.assertEqual(row["epoch_eval_splits"], ["train", "val"])
            self.assertFalse(row["evaluate_test_after_fit"])
            self.assertEqual(row["expected_train_id_hash"], row["expected_normalization_id_hash"])
            self.assertGreaterEqual(row["expected_pool_n"], 4000)
            self.assertGreater(row["expected_oof_n"], 0)
            self.assertGreater(row["expected_inner_val_n"], 0)
            self.assertNotEqual(row["expected_oof_id_hash"], row["expected_inner_val_id_hash"])
            if row["downsample_n_label"] == "full":
                self.assertIsNone(row["train_size_n"])
                self.assertEqual(row["subset_replicate"], 0)
                self.assertEqual(row["expected_train_n"], row["expected_pool_n"])
            else:
                self.assertEqual(row["expected_train_n"], row["train_size_n"])

    def test_pilot_and_command_families_parse(self):
        pilot = self.rows[0]
        self.assertEqual(pilot["row"], 1)
        self.assertEqual(pilot["cell_id"], REQUIRED_PILOT_CELL)
        self.assertEqual(pilot["part_slug"], "enhancer")
        self.assertEqual(pilot["stage4_lane"], "primary")
        self.assertEqual(pilot["downsample_n_label"], "40")

        representatives = {}
        for row in self.rows:
            key = (row["part_slug"], row["architecture"], row["training_regime"], row["loss_mode"])
            representatives.setdefault(key, row)
        for key, row in representatives.items():
            tokens = shlex.split(row["train_command"])
            with self.subTest(command_family=key):
                result = subprocess.run(
                    [sys.executable, *tokens[1:], "--help"],
                    cwd=LEARN,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_runner_registry_is_stage4_only(self):
        self.assertEqual(
            STAGE4_RUNS_CSV,
            LEARN
            / 'outputs/hpo_runs/status/lib1_dedup_stage4_downsampling_july2026/'
            'stage4_runs.csv',
        )
        with self.assertRaisesRegex(RuntimeError, 'non-Stage4'):
            read_registry(LEARN / 'run_registry/runs.csv')

    def test_zero_valued_provenance_is_not_treated_as_missing(self):
        self.assertEqual(provenance_scalar_text(0), "0")
        self.assertEqual(provenance_scalar_text(False), "False")
        self.assertEqual(provenance_scalar_text(None), "")


if __name__ == "__main__":
    unittest.main()
