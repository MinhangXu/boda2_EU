import argparse
import unittest
from pathlib import Path

from src.learn.verify_lib1_dedup_utr3_targeted_hpo_manifest import validate


REPO = Path(__file__).resolve().parents[1]
LEARN = REPO / "src/learn"
PREFIX = LEARN / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026"
MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
CONFIGS = Path(str(PREFIX) + "__search_configs.jsonl")
SUMMARY = Path(str(PREFIX) + "__summary.json")
STAGE2 = LEARN / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"


@unittest.skipUnless(
    all(path.is_file() for path in (MANIFEST, CONFIGS, SUMMARY, STAGE2)),
    "generated targeted 3'UTR HPO artifacts are unavailable",
)
class TargetedUtr3HpoManifestContractTests(unittest.TestCase):
    def test_fixed_grid_full_oof_pairing_and_audit_isolation(self):
        result = validate(
            argparse.Namespace(
                manifest=MANIFEST,
                search_configs=CONFIGS,
                summary=SUMMARY,
                stage2_analysis_manifest=STAGE2,
            )
        )
        self.assertEqual(result["validation_status"], "passed")
        self.assertEqual(result["new_base_configs"], 24)
        self.assertEqual(result["training_cells"], 240)
        self.assertEqual(result["complete_oof_arms"], 48)
        self.assertEqual(result["fold_level_rc_pairs"], 120)
        self.assertFalse(result["audit_loader_instantiated"])
        self.assertFalse(result["audit_ids_materialized"])
        self.assertEqual(result["commands_executed"], 0)


if __name__ == "__main__":
    unittest.main()
