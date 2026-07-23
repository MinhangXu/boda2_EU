import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import lib1_dedup_stage2_analysis as analysis
from src.analysis.lib1_dedup_stage1_analysis import INTRON_MASKS


def concrete_mask(mask):
    replacements = {"R": "A", "H": "A", "K": "G", "Y": "C", "N": "A"}
    return "".join(replacements.get(base, base) for base in mask)


class Stage2MetricTests(unittest.TestCase):
    def test_raw_metrics_are_on_raw_columns_and_cod_is_not_pearson_squared(self):
        frame = pd.DataFrame(
            {
                "log2_RNA_DNA": [0.0, 1.0, 2.0, 3.0],
                "prediction_raw": [1.0, 2.0, 3.0, 4.0],
                # Deliberately unrelated processed values.
                "target_processed": [0.0, 0.0, 0.0, 0.0],
                "prediction_processed": [0.0, 0.0, 0.0, 0.0],
            }
        )
        metrics = analysis.raw_metrics(frame)
        self.assertAlmostEqual(metrics["pearson"], 1.0)
        self.assertAlmostEqual(metrics["spearman"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 1.0)
        self.assertAlmostEqual(metrics["mae"], 1.0)
        self.assertAlmostEqual(metrics["cod_r2"], 0.2)

    def test_intron_metrics_use_inferred_sensitivity_labels_and_calibration(self):
        mask1 = concrete_mask(INTRON_MASKS["mask1_specific"])
        mask2_not_1 = "GTC" + "A" * 75 + "AG"
        residual = "A" * 80
        self.assertEqual(len(mask1), 80)
        self.assertEqual(len(mask2_not_1), 80)

        rows = []
        for stratum_index, sequence in enumerate((mask1, mask2_not_1, residual)):
            for within in range(3):
                target = 10.0 * stratum_index + within
                rows.append(
                    {
                        "construct_id": f"{stratum_index}-{within}",
                        "intron_sequence": sequence,
                        "log2_RNA_DNA": target,
                        "prediction_raw": 2.0 * target + 1.0,
                    }
                )
        frame = analysis.assign_inferred_intron_subsets(
            pd.DataFrame(rows), "intron_sequence"
        ).rename(
            columns={
                "inferred_intron_subset": analysis.SENSITIVITY_STRATUM,
            }
        )
        summary, per_stratum = analysis.intron_sensitivity_metrics(frame)
        self.assertEqual(
            {row[analysis.SENSITIVITY_STRATUM] for row in per_stratum},
            set(analysis.STRATUM_ORDER),
        )
        self.assertAlmostEqual(summary["within_stratum_centered_pearson"], 1.0)
        self.assertAlmostEqual(summary["macro_stratum_pearson"], 1.0)
        self.assertAlmostEqual(summary["minimum_stratum_pearson"], 1.0)
        for row in per_stratum:
            self.assertEqual(
                row["sensitivity_label_status"],
                "inferred_sequence_mask_not_true_subset",
            )
            self.assertAlmostEqual(
                row["calibration_slope_observed_on_prediction"], 0.5
            )
            self.assertAlmostEqual(
                row["calibration_intercept_observed_on_prediction"], -0.5
            )

    def test_paired_rc_comparison_is_construct_paired(self):
        metadata = {
            "analysis_lane": "core_scratch",
            "challenger_family": "none",
            "config_origin": "stage1_selected",
            "training_regime": "scratch",
            "part_slug": "enhancer",
            "architecture": "ResNet1DRegressor",
            "base_config_id": "basecfg_test",
            "policy_id": "basecfg_test",
            "initialization": "scratch",
            "source_head": "",
            "unfreeze_scope": "",
            "input_policy": "neutral_pad216_v1",
        }
        target = np.arange(10, dtype=float)
        off = pd.DataFrame(
            {
                **{key: [value] * 10 for key, value in metadata.items()},
                "rc_mode": ["off"] * 10,
                "development_fold": np.repeat(np.arange(5), 2),
                "construct_id": [f"construct-{index}" for index in range(10)],
                "log2_RNA_DNA": target,
                "prediction_raw": target + 1.0,
            }
        )
        on = off.copy()
        on["rc_mode"] = "on"
        on["prediction_raw"] = target
        arms = {
            ("core_scratch", "enhancer", "basecfg_test", "off"): off,
            ("core_scratch", "enhancer", "basecfg_test", "on"): on,
        }
        summary, fold_summary, paired = analysis.compare_paired_rc(arms)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(fold_summary), 5)
        self.assertEqual(len(paired), 10)
        self.assertAlmostEqual(
            summary.iloc[0]["mean_paired_abs_error_delta_on_minus_off"], -1.0
        )
        self.assertAlmostEqual(
            summary.iloc[0]["rc_on_lower_abs_error_fraction"], 1.0
        )

        bad_on = on.copy()
        bad_on.loc[0, "construct_id"] = "different"
        with self.assertRaisesRegex(ValueError, "identical constructs"):
            analysis.compare_paired_rc(
                {
                    ("core_scratch", "enhancer", "basecfg_test", "off"): off,
                    ("core_scratch", "enhancer", "basecfg_test", "on"): bad_on,
                }
            )

    def test_fold_metrics_keep_pooled_and_fold_estimands_separate(self):
        metadata = {
            "analysis_lane": "core_scratch",
            "challenger_family": "none",
            "config_origin": "stage1_selected",
            "training_regime": "scratch",
            "part_slug": "enhancer",
            "architecture": "ResNet1DRegressor",
            "base_config_id": "basecfg_test",
            "policy_id": "basecfg_test",
            "initialization": "scratch",
            "source_head": "",
            "unfreeze_scope": "",
            "input_policy": "neutral_pad216_v1",
            "rc_mode": "off",
        }
        rows = []
        for fold in range(5):
            for within in range(3):
                rows.append(
                    {
                        **metadata,
                        "development_fold": fold,
                        "construct_id": f"{fold}-{within}",
                        "log2_RNA_DNA": float(within),
                        "prediction_raw": float(within + fold),
                    }
                )
        arm = pd.DataFrame(rows)
        scored = analysis.score_oof_folds(
            {("core_scratch", "enhancer", "basecfg_test", "off"): arm}
        )
        self.assertEqual(len(scored), 5)
        self.assertEqual(set(scored["development_fold"]), set(range(5)))
        np.testing.assert_allclose(scored["fold_pearson"], 1.0)
        self.assertTrue((scored["fold_n_constructs"] == 3).all())

    def test_intron_stratum_mean_baseline_is_fit_out_of_fold(self):
        rows = []
        for index in range(analysis.EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS):
            fold = index % 5
            stratum = analysis.STRATUM_ORDER[index % 3]
            stratum_index = analysis.STRATUM_ORDER.index(stratum)
            rows.append(
                {
                    "construct_id": f"intron-{index}",
                    "development_fold": fold,
                    analysis.SENSITIVITY_STRATUM: stratum,
                    "log2_RNA_DNA": 2.0 * stratum_index + 0.01 * fold + (index % 7) / 10.0,
                    "prediction_raw": -999.0,
                }
            )
        arm = pd.DataFrame(rows)
        summaries, predictions = analysis.score_intron_stratum_mean_baselines(
            {("core_scratch", "intron", "basecfg_test", "off"): arm}
        )
        self.assertEqual(
            set(summaries["baseline_type"]),
            {
                "fold_trained_stratum_mean",
                "development_oracle_stratum_mean",
            },
        )
        self.assertEqual(len(predictions), 2 * len(arm))

        fold_zero = predictions.loc[
            predictions["baseline_type"].eq("fold_trained_stratum_mean")
            & predictions["development_fold"].eq(0)
        ]
        source = arm.loc[~arm["development_fold"].eq(0)]
        expected = source.groupby(analysis.SENSITIVITY_STRATUM)[
            "log2_RNA_DNA"
        ].mean()
        np.testing.assert_allclose(
            fold_zero["prediction_raw"],
            fold_zero[analysis.SENSITIVITY_STRATUM].map(expected),
        )

    def test_intron_rc_summary_and_folds_include_sensitivity_deltas(self):
        metadata = {
            "analysis_lane": "core_scratch",
            "challenger_family": "none",
            "config_origin": "stage1_selected",
            "training_regime": "scratch",
            "part_slug": "intron",
            "architecture": "ResNet1DRegressor",
            "base_config_id": "basecfg_intron",
            "policy_id": "basecfg_intron",
            "initialization": "scratch",
            "source_head": "",
            "unfreeze_scope": "",
            "input_policy": "exact80_v1",
        }
        rows = []
        for fold in range(5):
            for stratum_index, stratum in enumerate(analysis.STRATUM_ORDER):
                for within in range(3):
                    target = 10.0 * stratum_index + within
                    rows.append(
                        {
                            **metadata,
                            "development_fold": fold,
                            "construct_id": f"{fold}-{stratum}-{within}",
                            analysis.SENSITIVITY_STRATUM: stratum,
                            "log2_RNA_DNA": target,
                            "prediction_raw": 10.0 * stratum_index + (2 - within),
                        }
                    )
        off = pd.DataFrame(rows)
        off["rc_mode"] = "off"
        on = off.copy()
        on["rc_mode"] = "on"
        on["prediction_raw"] = on["log2_RNA_DNA"]
        summary, folds, _ = analysis.compare_paired_rc(
            {
                ("core_scratch", "intron", "basecfg_intron", "off"): off,
                ("core_scratch", "intron", "basecfg_intron", "on"): on,
            }
        )
        row = summary.iloc[0]
        self.assertAlmostEqual(row["rc_off_within_stratum_centered_pearson"], -1.0)
        self.assertAlmostEqual(row["rc_on_within_stratum_centered_pearson"], 1.0)
        self.assertAlmostEqual(
            row["delta_rc_on_minus_off_within_stratum_centered_pearson"], 2.0
        )
        self.assertAlmostEqual(row["delta_rc_on_minus_off_macro_stratum_pearson"], 2.0)
        self.assertAlmostEqual(
            row["delta_rc_on_minus_off_minimum_stratum_pearson"], 2.0
        )
        self.assertEqual(len(folds), 5)
        np.testing.assert_allclose(
            folds["delta_rc_on_minus_off_within_stratum_centered_pearson"], 2.0
        )
        self.assertEqual(
            row[
                "negative_fold_count_rc_on_minus_off_within_stratum_centered_pearson"
            ],
            0,
        )
        self.assertTrue(
            row["formal_pearson_fold_gate_mean_ge_0p005_and_positive_ge_4"]
        )
        self.assertTrue(row["formal_intron_pooled_and_within_fold_gate"])

    def test_launch_registry_identity_mismatch_is_rejected(self):
        manifest = {
            "execution_disposition": "launch",
            "planned_run_name": "stage2-cell",
            "cell_id": "expected-cell",
            "rc_pair_id": "pair-1",
            "analysis_lane": "core_scratch",
            "part_slug": "enhancer",
            "base_config_id": "basecfg_test",
            "development_fold": 2,
            "rc_mode": "on",
        }
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction.tsv"
            prediction.write_text("placeholder\n", encoding="utf-8")
            registry = Path(tmp) / "runs.csv"
            record = {
                "timestamp": "2026-07-12T00:00:00",
                "run_id": "run1",
                "run_name": "stage2-cell",
                "campaign_id": analysis.CAMPAIGN_ID,
                "campaign_stage": analysis.CAMPAIGN_STAGE,
                "status": "completed",
                "prediction_path": str(prediction),
                "cell_id": "wrong-cell",
                "rc_pair_id": "pair-1",
                "analysis_lane": "core_scratch",
                "part_slug": "enhancer",
                "base_config_id": "basecfg_test",
                "development_fold": "2",
                "rc_mode": "on",
            }
            with registry.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(record))
                writer.writeheader()
                writer.writerow(record)
            with self.assertRaisesRegex(ValueError, "wrong cell provenance"):
                analysis.resolve_analysis_cells([manifest], registry)


if __name__ == "__main__":
    unittest.main()
