import math
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import lib1_dedup_stage4_downsampling_analysis as analysis


class Stage4CurveTests(unittest.TestCase):
    def test_fisher_z_curves_are_bounded_monotone_and_report_disagreement(self):
        n = np.asarray([40, 250, 400, 2500, 4000, 6500], dtype=float)
        z = 0.9 - 2.2 * n ** -0.35
        pearson = np.tanh(z)
        records = []
        for family in ("power_law", "exponential"):
            fit = analysis.fit_saturating_curve(n, pearson, family, "pearson")
            self.assertEqual(fit["fit_status"], "success")
            predictions = analysis.predict_curve(fit, [40, 400, 4000, 40000])
            self.assertTrue(np.all(np.diff(predictions) >= -1e-12))
            self.assertTrue(np.all(np.abs(predictions) <= 1.0))
            records.append(
                {
                    "part_slug": "promoter",
                    "stage4_lane": "primary",
                    "base_config_id": "cfg",
                    "loo_rmse": 0.01,
                    **fit,
                }
            )
        disagreement = analysis.curve_family_disagreement(pd.DataFrame(records))
        self.assertEqual(len(disagreement), 1)
        self.assertGreaterEqual(disagreement.iloc[0]["absolute_10x_gain_disagreement"], 0)

    def test_rmse_curve_has_nonnegative_asymptote_and_decreases(self):
        n = np.asarray([40, 250, 400, 2500, 4000, 6500], dtype=float)
        rmse = 0.4 + 2.0 * n ** -0.3
        fit = analysis.fit_saturating_curve(n, rmse, "power_law", "rmse")
        self.assertEqual(fit["fit_status"], "success")
        prediction = analysis.predict_curve(fit, [40, 400, 4000, 40000])
        self.assertTrue(np.all(np.diff(prediction) <= 1e-12))
        self.assertGreaterEqual(fit["asymptote"], 0)


class Stage4PairedEvidenceTests(unittest.TestCase):
    @staticmethod
    def tracks(part="promoter"):
        rng = np.random.default_rng(7)
        n_constructs = 60 if part == "intron" else 50
        target = rng.normal(size=n_constructs)
        shared_noise = rng.normal(size=n_constructs)
        fold = np.repeat(np.arange(5), n_constructs // 5)
        ids = [f"id-{index:03d}" for index in range(n_constructs)]
        tracks = {}
        for n, noise_scale in ((40, 1.2), (400, 0.65), (4000, 0.25)):
            columns = {
                "outer_oof_fold": fold,
                "construct_id": ids,
                analysis.RAW_TARGET: target,
                analysis.RAW_PREDICTION: target + noise_scale * shared_noise,
            }
            if part == "intron":
                columns[analysis.SENSITIVITY_STRATUM] = np.resize(
                    np.asarray(analysis.STRATUM_ORDER), n_constructs
                )
            frame = pd.DataFrame(columns)
            frame.attrs["rows"] = [
                {
                    "part_slug": part,
                    "stage4_lane": "primary",
                    "diagnostic_only": False,
                    "portfolio_rank": 1,
                    "base_config_id": "cfg",
                    "architecture": "Synthetic",
                    "training_regime": "scratch",
                    "initialization": "scratch",
                    "source_head": "",
                    "unfreeze_scope": "all",
                    "rc_mode": "off",
                    "loss_mode": "unweighted_mse",
                    "policy_id": "synthetic",
                    "downsample_n_label": str(n),
                    "subset_replicate": 1,
                    "train_subsample_seed": 104729,
                    "model_seed": 1701,
                    "expected_train_n": n,
                    "outer_oof_fold": outer,
                }
                for outer in range(5)
            ]
            key = (part, "primary", "cfg", str(n), 1, 104729)
            tracks[key] = frame
        return tracks

    def test_observed_decade_contrasts_are_exactly_oof_paired(self):
        tracks = self.tracks()
        pooled, _ = analysis.score_pooled_tracks(tracks)
        detail, summary = analysis.observed_paired_contrasts(tracks, pooled)
        observed = detail.loc[
            detail["low_n"].eq(40) & detail["high_n"].eq(400)
        ].iloc[0]
        self.assertEqual(observed["paired_oof_constructs"], 50)
        self.assertGreater(observed["delta_pearson"], 0)
        self.assertEqual(len(summary), 3)  # 40->400, 400->4000, 40->4000

    def test_paired_bootstrap_is_deterministic_and_configurable(self):
        tracks = self.tracks()
        first = analysis.paired_bootstrap(tracks, resamples=8, seed=123)
        second = analysis.paired_bootstrap(tracks, resamples=8, seed=123)
        for left, right in zip(first, second):
            pd.testing.assert_frame_equal(left, right)
        contrast = first[1]
        self.assertFalse(contrast.empty)
        self.assertTrue((contrast["bootstrap_resamples"] == 8).all())

    def test_intron_bootstrap_includes_centered_and_each_frozen_stratum(self):
        products = analysis.paired_bootstrap(
            self.tracks(part="intron"), resamples=8, seed=123
        )
        points, contrasts, _, _, disagreement = products
        self.assertIn("within_stratum_centered", set(points["metric_scope"]))
        per_stratum = points.loc[points["metric_scope"].eq("per_stratum")]
        self.assertEqual(
            set(per_stratum[analysis.SENSITIVITY_STRATUM]),
            set(analysis.STRATUM_ORDER),
        )
        self.assertEqual(set(per_stratum["metric"]), set(analysis.METRICS))
        self.assertIn("within_stratum_centered", set(contrasts["metric_scope"]))
        self.assertEqual(
            set(contrasts.loc[
                contrasts["metric_scope"].eq("per_stratum"),
                analysis.SENSITIVITY_STRATUM,
            ]),
            set(analysis.STRATUM_ORDER),
        )
        self.assertFalse(disagreement.empty)
        self.assertTrue(
            (disagreement["successful_bootstrap_replicates"] <= 8).all()
        )


class Stage4TrainingDiagnosticTests(unittest.TestCase):
    @staticmethod
    def diagnostic_record():
        return {
            "checkpoint_monitor": "val_pearson",
            "best_metric_name": "val_pearson",
            "best_epoch": "12",
            "optimizer_steps": "137",
            "train_pearson": "0.61",
            "best_metric_value": "0.49",
            # The last-epoch summary is deliberately different and must not
            # be mistaken for the selected checkpoint's inner-val Pearson.
            "val_pearson": "0.31",
        }

    def test_gap_uses_best_checkpoint_metric_not_last_epoch_summary(self):
        record = self.diagnostic_record()
        payload = dict(record)
        evidence = analysis._training_diagnostic_evidence(record, payload)
        self.assertEqual(evidence["optimizer_steps"], 137)
        self.assertAlmostEqual(evidence["best_inner_val_pearson"], 0.49)
        self.assertAlmostEqual(
            evidence["train_minus_best_inner_val_pearson_gap"], 0.12
        )

    def test_completed_evidence_requires_optimizer_steps_in_both_products(self):
        record = self.diagnostic_record()
        payload = dict(record)
        del payload["optimizer_steps"]
        with self.assertRaisesRegex(RuntimeError, "optimizer_steps"):
            analysis._training_diagnostic_evidence(record, payload)


class Stage4IsolationTests(unittest.TestCase):
    def test_only_dedicated_stage4_registry_is_authorized(self):
        self.assertEqual(
            analysis.validate_registry_isolation(analysis.DEFAULT_REGISTRY),
            analysis.DEFAULT_REGISTRY.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "global runs.csv is forbidden"):
            analysis.validate_registry_isolation(
                analysis.LEARN_ROOT / "run_registry/runs.csv"
            )

    def test_dedicated_registry_rejects_foreign_and_final_test_rows(self):
        fields = [
            "campaign_id", "campaign_stage", "cell_id", *analysis.TEST_METRIC_FIELDS
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stage4_runs.csv"
            for row, message in (
                (
                    {
                        "campaign_id": "another_campaign",
                        "campaign_stage": analysis.EXPECTED_CAMPAIGN_STAGE,
                        "cell_id": "cell",
                    },
                    "another campaign",
                ),
                (
                    {
                        "campaign_id": analysis.EXPECTED_CAMPAIGN_ID,
                        "campaign_stage": analysis.EXPECTED_CAMPAIGN_STAGE,
                        "cell_id": "cell",
                        "test_pearson": "0.4",
                    },
                    "final-test metrics",
                ),
            ):
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow(row)
                with self.assertRaisesRegex(RuntimeError, message):
                    analysis.read_registry(path)

    def test_non_oof_and_final_test_paths_are_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "non-OOF"):
                analysis._assert_oof_path(root / "run__test_predictions.tsv")
            final = root / "final_test" / "run__oof_predictions.tsv"
            with self.assertRaisesRegex(ValueError, "final-test product tree"):
                analysis._assert_oof_path(final)

    def test_metric_definition_uses_raw_cod_not_squared_pearson(self):
        target = np.asarray([0.0, 1.0, 2.0, 3.0])
        prediction = np.asarray([0.0, 2.0, 4.0, 6.0])
        values = analysis._metrics_from_arrays(target, prediction)
        self.assertAlmostEqual(values["pearson"], 1.0)
        self.assertLess(values["cod_r2"], 0.0)
        self.assertTrue(math.isfinite(values["rmse"]))


if __name__ == "__main__":
    unittest.main()
