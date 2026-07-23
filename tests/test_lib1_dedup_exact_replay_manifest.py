import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.learn import generate_lib1_dedup_exact_replay_manifest as replay


REPO_ROOT = Path(__file__).resolve().parents[1]
LEARN_DIR = REPO_ROOT / "src" / "learn"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@unittest.skipUnless(
    (LEARN_DIR / "wandb" / "sweep-az1dlbv1").is_dir(),
    "historical W&B replay caches are unavailable",
)
class Lib1ExactReplayManifestTests(unittest.TestCase):
    def test_full_manifest_is_exact_unique_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tag = "stage1_manifest_contract_test"
            command = [
                sys.executable,
                str(LEARN_DIR / "generate_lib1_dedup_exact_replay_manifest.py"),
                "--manifest-tag",
                tag,
                "--outdir",
                tmp,
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
            path = Path(tmp) / (tag + "__run_manifest.jsonl")
            first_hash = file_sha256(path)
            subprocess.run(command, cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
            self.assertEqual(first_hash, file_sha256(path))

            with path.open() as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(
                Counter(row["run_kind"] for row in rows),
                Counter({"exact_replay": 885, "pre_dedup_calibration": 25}),
            )
            exact = [row for row in rows if row["run_kind"] == "exact_replay"]
            self.assertEqual(len({row["base_config_id"] for row in exact}), 885)
            self.assertEqual(len({row["planned_run_name"] for row in rows}), 910)
            self.assertEqual(len({row["row_fingerprint"] for row in rows}), 910)
            self.assertEqual(
                Counter(row["part_slug"] for row in exact),
                Counter(
                    enhancer=128,
                    promoter=158,
                    intron=156,
                    utr3=157,
                    utr5=286,
                ),
            )
            for row in exact:
                self.assertEqual(row["development_fold"], 0)
                self.assertEqual(row["model_seed"], 1701)
                self.assertFalse(row["use_reverse_complements"])
                self.assertFalse(row["barcode_weighting"])
                self.assertFalse(row["evaluate_test_after_fit"])
                self.assertEqual(row["artifact_retention"], "none")
                self.assertEqual(row["epoch_eval_splits"], ["train", "val"])
                self.assertEqual(row["prediction_splits"], ["val"])
                self.assertEqual(row["wandb_entity"], replay.EXPECTED_ENTITY)
                tokens = shlex.split(row["train_command"])
                self.assertNotIn("test", tokens[tokens.index("--epoch_eval_splits") + 1 :])

    def test_inactive_scheduler_fields_do_not_change_base_identity(self):
        left = {
            "model_module": "ResNet1DRegressor",
            "scheduler": "None",
            "T_0": 10,
            "T_mult": 7,
            "eta_min": 0.5,
        }
        right = dict(left, T_0=1000, T_mult=1, eta_min=0)
        self.assertEqual(replay.base_identity(left), replay.base_identity(right))


if __name__ == "__main__":
    unittest.main()
