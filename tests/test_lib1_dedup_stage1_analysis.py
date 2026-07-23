import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis import lib1_dedup_stage1_analysis as analysis


class Stage1ResultLoadingTests(unittest.TestCase):
    def test_manifest_part_slug_survives_blank_newer_registry_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            manifest_row = {
                "manifest_row": 1,
                "run_kind": "exact_replay",
                "planned_run_name": "stage1-run",
                "part_slug": "intron",
                "lane_id": "intron__resnet1d",
                "architecture": "ResNet1DRegressor",
                "base_config_id": "basecfg_test",
                "base_config_sha256": "test",
                "row_fingerprint": "fingerprint",
                "dataset_path": str(root / "dataset.tsv"),
                "split_manifest_path": str(root / "split.json"),
                "source_candidates": [{"candidate_kind": "completed_hpo"}],
                "base_identity": {"lr": 0.001},
            }
            manifest.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

            registry = root / "runs.csv"
            record = {
                "timestamp": "2026-07-13T00:00:00",
                "run_id": "run1",
                "run_name": "stage1-run",
                "campaign_id": analysis.CAMPAIGN_ID,
                "campaign_stage": analysis.EXACT_STAGE,
                # Stage 1 predates this registry field; an appended schema leaves it blank.
                "part_slug": "",
                "prediction_path": "",
            }
            for column in analysis.NUMERIC_COLUMNS:
                record[column] = "0"
            with registry.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerow(record)

            loaded = analysis.load_stage1_results(manifest, registry)
            self.assertEqual(loaded.loc[0, "part_slug"], "intron")


if __name__ == "__main__":
    unittest.main()
