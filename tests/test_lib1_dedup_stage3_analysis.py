import unittest

import numpy as np
import pandas as pd

from src.analysis import lib1_dedup_stage3_analysis as analysis


def synthetic_arm(
    *,
    part="enhancer",
    config="basecfg_test",
    rc_mode="off",
    loss_mode="unweighted_mse",
    noise=0.5,
):
    rows = []
    for fold in analysis.EXPECTED_FOLDS:
        if part == "intron":
            iterator = [
                (stratum, within)
                for stratum in analysis.STRATUM_ORDER
                for within in range(5)
            ]
        else:
            iterator = [("", within) for within in range(15)]
        for index, (stratum, within) in enumerate(iterator):
            stratum_index = (
                analysis.STRATUM_ORDER.index(stratum) if part == "intron" else 0
            )
            target = 4.0 * stratum_index + float(within)
            prediction = target + noise * (-1.0 if index % 2 else 1.0)
            row = {
                "part_slug": part,
                "portfolio_rank": 1,
                "portfolio_role": "test",
                "base_config_id": config,
                "architecture": "ResNet1DRegressor",
                "analysis_lane": "test",
                "training_regime": "scratch",
                "initialization": "scratch",
                "source_head": "",
                "unfreeze_scope": "",
                "input_policy": "test",
                "policy_id": config,
                "rc_mode": rc_mode,
                "loss_mode": loss_mode,
                "development_fold": fold,
                "construct_id": f"{fold}-{stratum}-{within}",
                analysis.RAW_TARGET: target,
                analysis.RAW_PREDICTION: prediction,
                "cell_id": f"cell-{fold}-{rc_mode}-{loss_mode}",
                "loss_pair_id": f"loss-pair-{fold}-{rc_mode}",
                "rc_pair_id": f"rc-pair-{fold}-{loss_mode}",
            }
            if part == "intron":
                row[analysis.SENSITIVITY_STRATUM] = stratum
            rows.append(row)
    return pd.DataFrame(rows)


class Stage3GateTests(unittest.TestCase):
    margin = {
        "allowed_pooled_rmse_increase": 0.01,
        "allowed_pooled_cod_r2_decrease": 0.01,
        "numeric_epsilon": 1e-12,
    }

    def test_improving_weighted_arm_passes_exact_five_fold_gate(self):
        baseline = synthetic_arm(noise=0.8)
        weighted = synthetic_arm(loss_mode="barcode_weighted_mse", noise=0.05)
        summary, folds = analysis.paired_gate(
            baseline=baseline,
            intervention=weighted,
            part="enhancer",
            base_config_id="basecfg_test",
            gate_kind="weighted_minus_unweighted",
            margin=self.margin,
            pair_id_column="loss_pair_id",
        )
        self.assertEqual(len(folds), 5)
        self.assertTrue(summary["all_five_fold_pearson_deltas_finite"])
        self.assertEqual(summary["positive_fold_pearson_delta_count"], 5)
        self.assertTrue(summary["gate_pass"])

    def test_a_nonfinite_fold_delta_is_not_dropped_and_fails_gate(self):
        baseline = synthetic_arm(noise=0.8)
        baseline.loc[baseline["development_fold"].eq(2), analysis.RAW_PREDICTION] = 1.0
        weighted = synthetic_arm(loss_mode="barcode_weighted_mse", noise=0.05)
        summary, folds = analysis.paired_gate(
            baseline=baseline,
            intervention=weighted,
            part="enhancer",
            base_config_id="basecfg_test",
            gate_kind="weighted_minus_unweighted",
            margin=self.margin,
            pair_id_column="loss_pair_id",
        )
        self.assertEqual(len(folds), 5)
        self.assertEqual(summary["finite_fold_delta_count"], 4)
        self.assertFalse(summary["all_five_fold_pearson_deltas_finite"])
        self.assertFalse(summary["gate_pass"])


class Stage3AdmissibilityTests(unittest.TestCase):
    def test_rc_on_weighted_requires_both_intervention_gates(self):
        metric_rows = []
        loss_rows = []
        rc_rows = []
        for part in analysis.PART_ORDER:
            for config_index in range(10):
                config = f"basecfg_{part}_{config_index}"
                rc_modes = ("off",) if part == "utr3" else ("off", "on")
                for rc_mode in rc_modes:
                    for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
                        metric_rows.append(
                            {
                                "part_slug": part,
                                "base_config_id": config,
                                "rc_mode": rc_mode,
                                "loss_mode": loss_mode,
                                "pooled_oof_pearson": 0.5,
                                "minimum_fold_pearson": 0.4,
                                "pooled_oof_rmse": 0.5,
                                "pooled_oof_cod_r2": 0.2,
                                "minimum_stratum_pearson": 0.3,
                                "within_stratum_centered_pearson": 0.35,
                            }
                        )
                    loss_rows.append(
                        {
                            "part_slug": part,
                            "base_config_id": config,
                            "baseline_rc_mode": rc_mode,
                            "gate_pass": not (
                                part == "enhancer"
                                and config_index == 0
                                and rc_mode == "on"
                            ),
                        }
                    )
                if part != "utr3":
                    for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
                        rc_rows.append(
                            {
                                "part_slug": part,
                                "base_config_id": config,
                                "baseline_loss_mode": loss_mode,
                                "gate_pass": not (
                                    part == "promoter"
                                    and config_index == 0
                                    and loss_mode == "barcode_weighted_mse"
                                ),
                            }
                        )
        result = analysis.apply_admissibility(
            pd.DataFrame(metric_rows), pd.DataFrame(loss_rows), pd.DataFrame(rc_rows)
        )
        self.assertEqual(len(result), 180)
        enhancer = result.loc[
            result["base_config_id"].eq("basecfg_enhancer_0")
            & result["rc_mode"].eq("on")
            & result["loss_mode"].eq("barcode_weighted_mse")
        ].iloc[0]
        self.assertFalse(enhancer["admissible"])
        promoter = result.loc[
            result["base_config_id"].eq("basecfg_promoter_0")
            & result["rc_mode"].eq("on")
            & result["loss_mode"].eq("barcode_weighted_mse")
        ].iloc[0]
        self.assertFalse(promoter["admissible"])
        baselines = result.loc[
            result["rc_mode"].eq("off")
            & result["loss_mode"].eq("unweighted_mse")
        ]
        self.assertEqual(len(baselines), 50)
        self.assertTrue(baselines["admissible"].all())

    def test_nonfinite_required_selection_metric_makes_arm_ineligible(self):
        rows = []
        loss_rows = []
        rc_rows = []
        for part in analysis.PART_ORDER:
            for config_index in range(10):
                config = f"basecfg_{part}_{config_index}"
                rc_modes = ("off",) if part == "utr3" else ("off", "on")
                for rc_mode in rc_modes:
                    for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
                        rows.append(
                            {
                                "part_slug": part,
                                "base_config_id": config,
                                "rc_mode": rc_mode,
                                "loss_mode": loss_mode,
                                "pooled_oof_pearson": 0.5,
                                "minimum_fold_pearson": (
                                    np.nan
                                    if part == "utr3"
                                    and config_index == 0
                                    and rc_mode == "off"
                                    and loss_mode == "unweighted_mse"
                                    else 0.4
                                ),
                                "pooled_oof_rmse": 0.5,
                                "pooled_oof_cod_r2": 0.2,
                                "minimum_stratum_pearson": 0.3,
                                "within_stratum_centered_pearson": 0.35,
                            }
                        )
                    loss_rows.append(
                        {
                            "part_slug": part,
                            "base_config_id": config,
                            "baseline_rc_mode": rc_mode,
                            "gate_pass": True,
                        }
                    )
                if part != "utr3":
                    for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
                        rc_rows.append(
                            {
                                "part_slug": part,
                                "base_config_id": config,
                                "baseline_loss_mode": loss_mode,
                                "gate_pass": True,
                            }
                        )
        result = analysis.apply_admissibility(
            pd.DataFrame(rows), pd.DataFrame(loss_rows), pd.DataFrame(rc_rows)
        )
        arm = result.loc[
            result["base_config_id"].eq("basecfg_utr3_0")
            & result["loss_mode"].eq("unweighted_mse")
        ].iloc[0]
        self.assertFalse(arm["admissible"])
        self.assertFalse(arm["selection_eligible"])
        self.assertEqual(
            arm["admissibility_reason"],
            "nonfinite_required_selection_metric",
        )
        self.assertEqual(
            arm["selection_ineligibility_reason"],
            "nonfinite_required_selection_metric",
        )

    def test_undefined_intron_stratum_pearson_propagates_to_minimum(self):
        frame = synthetic_arm(part="intron", noise=0.4)
        constant = frame[analysis.SENSITIVITY_STRATUM].eq(
            analysis.STRATUM_ORDER[0]
        )
        frame.loc[constant, analysis.RAW_PREDICTION] = 1.0
        metrics, _, strata, _ = analysis.score_arms(
            {("intron", "basecfg_test", "off", "unweighted_mse"): frame},
            {"basecfg_test": 1},
        )
        self.assertTrue(np.isnan(metrics.iloc[0]["minimum_stratum_pearson"]))
        observed = strata.set_index(analysis.SENSITIVITY_STRATUM)["pearson"]
        self.assertTrue(np.isnan(observed.loc[analysis.STRATUM_ORDER[0]]))


class Stage3SelectionAndFactorialTests(unittest.TestCase):
    def test_bootstrap_is_sort_invariant_and_rng_is_reinitialized(self):
        frame = synthetic_arm(noise=0.4)
        shuffled = frame.sample(frac=1.0, random_state=17)
        first, metadata = analysis.bootstrap_best_arm(
            frame, part="enhancer", resamples=50
        )
        second, _ = analysis.bootstrap_best_arm(
            shuffled, part="promoter", resamples=50
        )
        np.testing.assert_allclose(first, second)
        self.assertTrue(metadata["rng_reinitialized_for_part"])
        self.assertEqual(
            metadata["bootstrap_within_fold_sort"],
            "development_fold_then_construct_id_ascending",
        )

    def test_exact_tie_complexity_rule_is_part_specific(self):
        common = {
            "minimum_fold_pearson": 0.4,
            "pooled_oof_rmse": 0.5,
            "pooled_oof_cod_r2": 0.2,
            "rc_mode": "off",
            "loss_mode": "unweighted_mse",
        }
        transfer_only = pd.DataFrame(
            [
                {
                    **common,
                    "base_config_id": "full",
                    "training_regime": "transfer",
                    "unfreeze_scope": "full",
                    "model_parameter_count": 1,
                },
                {
                    **common,
                    "base_config_id": "branched",
                    "training_regime": "transfer",
                    "unfreeze_scope": "branched_only",
                    "model_parameter_count": 100,
                },
            ]
        )
        ordered = analysis.order_one_se_band(transfer_only, "enhancer")
        self.assertEqual(ordered.iloc[0]["base_config_id"], "branched")

        mixed = pd.concat(
            [
                transfer_only.iloc[[0]],
                pd.DataFrame(
                    [
                        {
                            **common,
                            "base_config_id": "scratch",
                            "training_regime": "scratch",
                            "unfreeze_scope": "",
                            "model_parameter_count": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        ordered_mixed = analysis.order_one_se_band(mixed, "enhancer")
        self.assertEqual(ordered_mixed.iloc[0]["base_config_id"], "scratch")

    def test_factorial_export_has_exact_40_by_5_accounting(self):
        arms = {}
        for part in ("enhancer", "promoter", "intron", "utr5"):
            for config_index in range(10):
                config = f"basecfg_{part}_{config_index}"
                for rc_mode in ("off", "on"):
                    for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
                        arms[(part, config, rc_mode, loss_mode)] = synthetic_arm(
                            part=part,
                            config=config,
                            rc_mode=rc_mode,
                            loss_mode=loss_mode,
                            noise=0.4,
                        )
        summaries, folds = analysis.score_factorial_differences(arms)
        self.assertEqual(len(summaries), 40)
        self.assertEqual(len(folds), 200)
        self.assertTrue(
            summaries["all_five_fold_pearson_interactions_finite"].all()
        )


if __name__ == "__main__":
    unittest.main()
