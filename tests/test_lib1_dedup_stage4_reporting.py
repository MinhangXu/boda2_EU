import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis import lib1_dedup_stage4_reporting as reporting


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class Stage4ReportingReadinessTests(unittest.TestCase):
    def _valid_directory(self, root: Path) -> None:
        _write_json(
            root / "stage4_readiness.json",
            {
                "analysis_mode": "completed_oof_only",
                "manifest_rows": 660,
                "completed_cells": 660,
                "remaining_cells": 0,
                "complete_pooled_tracks": 132,
                "curve_point_rows": 72,
                "bootstrap_resamples": 2000,
                "manifest_validation_status": "valid",
                "global_registry_read": False,
                "final_test_loader_instantiated": False,
                "final_test_products_read": False,
                "final_test_metrics_computed": False,
            },
        )
        _write_json(
            root / "stage4_analysis_contract.json",
            {
                "primary_estimand": "pooled_five_fold_development_oof_pearson",
                "bootstrap": {"resamples": 2000},
                "registry": {"global_registry_read": False},
                "final_test_loader_instantiated": False,
                "final_test_products_read": False,
            },
        )
        pd.DataFrame(
            {
                "cell_id": [f"cell_{index}" for index in range(660)],
                "availability": ["complete"] * 660,
            }
        ).to_csv(root / "stage4_completion.csv", index=False)

    def test_complete_isolated_core_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._valid_directory(root)
            readiness, contract = reporting.validate_core_readiness(root)
            self.assertEqual(readiness["completed_cells"], 660)
            self.assertFalse(contract["final_test_products_read"])

    def test_incomplete_campaign_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._valid_directory(root)
            payload = json.loads((root / "stage4_readiness.json").read_text())
            payload.update(completed_cells=659, remaining_cells=1)
            _write_json(root / "stage4_readiness.json", payload)
            with self.assertRaisesRegex(RuntimeError, "not 660/660 complete"):
                reporting.validate_core_readiness(root)

    def test_final_test_access_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._valid_directory(root)
            payload = json.loads((root / "stage4_readiness.json").read_text())
            payload["final_test_products_read"] = True
            _write_json(root / "stage4_readiness.json", payload)
            with self.assertRaisesRegex(RuntimeError, "final_test_products_read"):
                reporting.validate_core_readiness(root)

    def test_nonstandard_bootstrap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._valid_directory(root)
            payload = json.loads((root / "stage4_readiness.json").read_text())
            payload["bootstrap_resamples"] = 100
            _write_json(root / "stage4_readiness.json", payload)
            with self.assertRaisesRegex(RuntimeError, "2,000-replicate"):
                reporting.validate_core_readiness(root)


class Stage4ReportingTableTests(unittest.TestCase):
    def test_direct_100x_table_is_observed_and_paired(self) -> None:
        summary_rows = []
        bootstrap_rows = []
        for index, part in enumerate(reporting.PARTS):
            config = f"primary_{part}"
            delta = 0.10 + 0.01 * index
            common = {
                "part_slug": part,
                "stage4_lane": "primary",
                "base_config_id": config,
                "low_n": 40,
                "high_n": 4000,
                "multiplicative_contrast": "100x",
            }
            summary_rows.append({**common, "mean_delta_pearson": delta})
            bootstrap_rows.append(
                {
                    **common,
                    "metric": "pearson",
                    "metric_scope": "overall",
                    "ci_2_5": delta - 0.02,
                    "ci_97_5": delta + 0.02,
                }
            )
        result = reporting.observed_100x_table(
            {
                "contrasts": pd.DataFrame(summary_rows),
                "boot_contrasts": pd.DataFrame(bootstrap_rows),
            }
        )
        self.assertEqual(len(result), 5)
        self.assertTrue(result["evidence"].eq("positive").all())
        self.assertTrue(result["contrast"].eq("40→4,000").all())

    def test_alternative_deltas_use_matching_subset_track(self) -> None:
        rows = []
        for part_index, part in enumerate(reporting.PARTS):
            alternative_count = 1 if part == "utr5" else 2
            for label_index, label in enumerate(("40", "400", "4000", "full")):
                primary_value = 0.20 + 0.01 * part_index + 0.02 * label_index
                rows.append(
                    {
                        "part_slug": part,
                        "stage4_lane": "primary",
                        "downsample_n_label": label,
                        "subset_replicate": 0 if label == "full" else 1,
                        "train_subsample_seed": 0 if label == "full" else 104729,
                        "mean_actual_train_n": 5000 if label == "full" else int(label),
                        "pearson": primary_value,
                        "portfolio_rank": 1,
                        "base_config_id": f"primary_{part}",
                        "architecture": "primary",
                    }
                )
                # A second primary subset must not be averaged into the match.
                if label != "full":
                    rows.append(
                        {
                            "part_slug": part,
                            "stage4_lane": "primary",
                            "downsample_n_label": label,
                            "subset_replicate": 2,
                            "train_subsample_seed": 130363,
                            "mean_actual_train_n": int(label),
                            "pearson": primary_value - 0.10,
                            "portfolio_rank": 1,
                            "base_config_id": f"primary_{part}",
                            "architecture": "primary",
                        }
                    )
                for alt_index in range(alternative_count):
                    rows.append(
                        {
                            "part_slug": part,
                            "stage4_lane": "alternative",
                            "downsample_n_label": label,
                            "subset_replicate": 0 if label == "full" else 1,
                            "train_subsample_seed": 0 if label == "full" else 104729,
                            "mean_actual_train_n": 5000 if label == "full" else int(label),
                            "pearson": primary_value + 0.01 * (alt_index + 1),
                            "portfolio_rank": alt_index + 2,
                            "base_config_id": f"alternative_{part}_{alt_index}",
                            "architecture": "alternative",
                        }
                    )
        result = reporting.alternative_point_deltas({"pooled": pd.DataFrame(rows)})
        self.assertEqual(len(result), 36)
        self.assertTrue(
            set(result["uncertainty_status"])
            == {"point_delta_only_no_paired_bootstrap_interval"}
        )
        first = result.loc[
            result["alternative_base_config_id"].eq("alternative_enhancer_0")
            & result["downsample_n_label"].eq("40")
        ].iloc[0]
        self.assertAlmostEqual(first["alternative_minus_primary_pearson"], 0.01)


if __name__ == "__main__":
    unittest.main()
