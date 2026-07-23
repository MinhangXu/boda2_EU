import unittest

import numpy as np
import pandas as pd

from src.analysis import lib1_dedup_stage2_reporting as reporting


class Stage2ReportingTests(unittest.TestCase):
    def test_arm_decision_table_reports_predeclared_fold_statistics(self):
        oof = pd.DataFrame(
            [
                {
                    "analysis_lane": "core_scratch",
                    "part_slug": "promoter",
                    "base_config_id": "basecfg_a",
                    "rc_mode": "off",
                    "pooled_oof_pearson": 0.5,
                }
            ]
        )
        folds = pd.DataFrame(
            [
                {
                    "analysis_lane": "core_scratch",
                    "part_slug": "promoter",
                    "base_config_id": "basecfg_a",
                    "rc_mode": "off",
                    "fold_pearson": value,
                }
                for value in (0.1, 0.2, 0.3, 0.4, 0.5)
            ]
        )
        result = reporting.arm_decision_table(oof, folds).iloc[0]
        self.assertAlmostEqual(result["fold_pearson_mean"], 0.3)
        self.assertAlmostEqual(result["fold_pearson_min"], 0.1)
        self.assertAlmostEqual(result["fold_pearson_p20"], 0.18)
        self.assertEqual(result["positive_fold_count"], 5)

    def test_strict_rc_gate_uses_mean_threshold_and_four_positive_folds(self):
        base = {
            "analysis_lane": "core_scratch",
            "part_slug": "promoter",
            "base_config_id": "basecfg_a",
        }
        rc = pd.DataFrame(
            [
                {
                    **base,
                    "delta_rc_on_minus_off_pooled_oof_rmse": -0.01,
                    "delta_rc_on_minus_off_pooled_oof_cod_r2": 0.01,
                    "mean_fold_delta_rc_on_minus_off_within_stratum_centered_pearson": np.nan,
                    "negative_fold_count_rc_on_minus_off_within_stratum_centered_pearson": np.nan,
                }
            ]
        )
        rc_folds = pd.DataFrame(
            [
                {**base, "delta_rc_on_minus_off_pooled_pearson": value}
                for value in (0.02, 0.02, 0.01, 0.01, -0.005)
            ]
        )
        result = reporting.strict_rc_table(rc, rc_folds).iloc[0]
        self.assertTrue(result["strict_pearson_fold_gate"])
        self.assertTrue(result["strict_gate_with_zero_tolerance_error_guard"])

    def test_utr5_diversity_review_preserves_pure_ranking(self):
        rows = []
        for rank in range(1, 7):
            architecture = "UTR_BassetVL" if rank <= 5 else "ResNet1DRegressor"
            for rc_mode, delta in (("off", 0.0), ("on", -0.1)):
                rows.append(
                    {
                        "analysis_lane": "core_scratch",
                        "part_slug": "utr5",
                        "base_config_id": f"basecfg_{rank}",
                        "rc_mode": rc_mode,
                        "architecture": architecture,
                        "policy_id": f"basecfg_{rank}",
                        "source_head": "",
                        "unfreeze_scope": "",
                        "input_policy": "exact50_v1",
                        "pooled_oof_pearson": 0.60 - rank * 0.01 + delta,
                        "pooled_oof_rmse": 0.4,
                    }
                )
        review = reporting.stage3_selection_review(pd.DataFrame(rows))
        self.assertEqual(int(review["pure_pooled_top5"].sum()), 5)
        recommended = review.loc[review["recommended_stage3_slot"]]
        self.assertEqual(len(recommended), 5)
        self.assertIn("ResNet1DRegressor", set(recommended["architecture"]))
        self.assertIn(
            "pure_rank5_diversity_alternate", set(review["selection_reason"])
        )


if __name__ == "__main__":
    unittest.main()
