import unittest

import numpy as np
import pandas as pd

from src.analysis import lib1_dedup_intron_sensitivity_reporting as reporting


def example_frame():
    rows = []
    specifications = (
        ("mask1_specific", 3.0, [0.0, 0.5, 1.0], "GT" + "A" * 76 + "AG"),
        ("mask2_not_mask1", 2.0, [1.0, 0.0, 0.5], "GT" + "C" * 76 + "AG"),
        ("mask3_residual", 1.0, [0.0, 1.0, 0.5], "A" * 80),
    )
    construct = 0
    for stratum, offset, predictions, sequence in specifications:
        for within, prediction_delta in enumerate(predictions):
            rows.append(
                {
                    "construct_id": "construct-{}".format(construct),
                    "development_fold": within,
                    "log2_RNA_DNA": offset + within * 0.5,
                    "prediction_raw": offset + prediction_delta,
                    "inferred_intron_sensitivity_stratum": stratum,
                    "sequence": sequence,
                    "n_barcodes": 8 + within,
                }
            )
            construct += 1
    return pd.DataFrame(rows)


class IntronSensitivityReportingTests(unittest.TestCase):
    def test_decomposition_reconstructs_population_moments(self):
        frame = example_frame()
        result = reporting.covariance_decomposition(frame).set_index("component")
        self.assertAlmostEqual(
            result.loc["target_variance", "total"],
            float(np.var(frame["log2_RNA_DNA"], ddof=0)),
        )
        self.assertAlmostEqual(
            result.loc["prediction_variance", "total"],
            float(np.var(frame["prediction_raw"], ddof=0)),
        )
        target = frame["log2_RNA_DNA"].to_numpy(float)
        prediction = frame["prediction_raw"].to_numpy(float)
        covariance = float(
            np.mean((target - target.mean()) * (prediction - prediction.mean()))
        )
        self.assertAlmostEqual(
            result.loc["target_prediction_covariance", "total"], covariance
        )

    def test_equal_stratum_metric_has_finite_effective_sample_size(self):
        frame = pd.concat(
            [example_frame(), example_frame().loc[lambda value: value.index < 2]],
            ignore_index=True,
        )
        frame["construct_id"] = ["row-{}".format(index) for index in range(len(frame))]
        result = reporting.intron_estimands(frame)
        self.assertTrue(np.isfinite(result["equal_stratum_pooled_pearson"]))
        self.assertGreater(result["equal_stratum_weight_ess"], 0)
        self.assertLessEqual(result["equal_stratum_weight_ess"], len(frame))

    def test_literal_equal_bases_conflict_with_equal_strata(self):
        result = reporting.literal_base_balance_constraints(example_frame())
        self.assertEqual(set(result["position_1_based"]), {1, 2, 79, 80})
        self.assertTrue(
            (result["maximum_structured_mass_if_fixed_base_frequency_is_0p25"] == 0.25).all()
        )
        self.assertFalse(
            result["literal_equal_bases_compatible_with_equal_strata"].any()
        )
        self.assertAlmostEqual(
            result.iloc[0]["natural_structured_stratum_mass"], 2.0 / 3.0
        )

    def test_literal_position_balance_lp_reports_infeasible_support(self):
        result = reporting.literal_position_balance_linear_program(
            example_frame()
        ).iloc[0]
        self.assertFalse(result["exact_25pct_each_base_each_position_feasible"])
        self.assertGreater(
            result["minimum_max_absolute_marginal_deviation"], 0
        )
        self.assertGreaterEqual(
            result["maximum_residual_mass_at_optimum"],
            result["minimum_residual_mass_at_optimum"],
        )

    def test_barcode_thresholds_are_descriptive_subsets(self):
        result = reporting.barcode_threshold_sensitivity(
            example_frame(), thresholds=(8, 9)
        )
        self.assertEqual(result["n_constructs"].tolist(), [9.0, 6.0])
        self.assertEqual(
            set(result["analysis_status"]),
            {"development_only_post_stage2_descriptive"},
        )


if __name__ == "__main__":
    unittest.main()
