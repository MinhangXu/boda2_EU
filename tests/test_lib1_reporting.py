import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.lib1_reporting import (
    assert_exact_levels,
    assert_paired_keys,
    assert_unique_keys,
    comparison_subplots,
    find_repo_root,
    harmonize_y_limits,
    load_analysis_bundle,
    require_columns,
    save_figure,
    sha256_file,
)


class ReportingContractTests(unittest.TestCase):
    def test_repo_root_and_dataframe_contracts(self):
        repo = Path(__file__).resolve().parents[1]
        self.assertEqual(find_repo_root(repo / "tutorials/lib1_tasks"), repo)

        frame = pd.DataFrame(
            {
                "config": ["a", "a", "b", "b"],
                "rc_mode": ["off", "on", "off", "on"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        require_columns(frame, ["config", "value"], "metrics")
        assert_unique_keys(frame, ["config", "rc_mode"], "metrics")
        assert_exact_levels(frame, "rc_mode", ["off", "on"], "metrics")
        assert_paired_keys(frame, ["config"], "rc_mode", table_name="metrics")

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            require_columns(frame, ["missing"], "metrics")
        with self.assertRaisesRegex(ValueError, "duplicate rows"):
            assert_unique_keys(pd.concat([frame, frame.iloc[[0]]]), ["config", "rc_mode"])
        with self.assertRaisesRegex(ValueError, "does not contain exactly"):
            assert_paired_keys(frame.iloc[:-1], ["config"], "rc_mode")

    def test_analysis_bundle_validates_summary_and_hashes_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / "summary.json"
            summary.write_text(
                json.dumps({"analysis_cells": 660, "audit_loader_instantiated": False}),
                encoding="utf-8",
            )
            metrics = root / "metrics.csv"
            metrics.write_text("config,value\na,0.5\n", encoding="utf-8")
            pairs = root / "pairs.tsv.gz"
            pd.DataFrame({"config": ["a"], "delta": [0.1]}).to_csv(
                pairs, sep="\t", index=False, compression="gzip"
            )

            bundle = load_analysis_bundle(
                root,
                {"metrics": "metrics.csv", "pairs": "pairs.tsv.gz"},
                expected_summary={
                    "analysis_cells": 660,
                    "audit_loader_instantiated": False,
                },
                summary_file="summary.json",
            )
            self.assertEqual(bundle.table("metrics").shape, (1, 2))
            self.assertEqual(bundle.table("pairs").shape, (1, 2))
            self.assertEqual(bundle.sha256["metrics"], sha256_file(metrics))
            self.assertEqual(set(bundle.paths), {"summary", "metrics", "pairs"})

            with self.assertRaisesRegex(ValueError, "summary contract differs"):
                load_analysis_bundle(
                    root,
                    {"metrics": "metrics.csv"},
                    expected_summary={"analysis_cells": 659},
                    summary_file="summary.json",
                )


class ReportingPlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_multi_panel_requires_explicit_comparability_policy(self):
        with self.assertRaisesRegex(ValueError, "explicit y_groups policy"):
            comparison_subplots(1, 2)

    def test_row_groups_share_only_within_each_metric_group(self):
        fig, axes = comparison_subplots(2, 2, y_groups="row", figsize=(6, 4))
        axes[0, 0].plot([0, 1], [0, 1])
        axes[0, 1].plot([0, 1], [10, 20])
        axes[1, 0].plot([0, 1], [-100, -80])
        axes[1, 1].plot([0, 1], [-60, -40])
        harmonize_y_limits(axes, pad_fraction=0.1)

        self.assertEqual(axes[0, 0].get_ylim(), axes[0, 1].get_ylim())
        self.assertEqual(axes[1, 0].get_ylim(), axes[1, 1].get_ylim())
        self.assertNotEqual(axes[0, 0].get_ylim(), axes[1, 0].get_ylim())
        self.assertLessEqual(axes[0, 0].get_ylim()[0], 0.0)
        self.assertGreaterEqual(axes[0, 0].get_ylim()[1], 20.0)
        plt.close(fig)

    def test_explicit_groups_and_empty_axes_are_safe(self):
        fig, axes = comparison_subplots(
            1, 3, y_groups=[[(0, 0), (0, 2)], [(0, 1)]]
        )
        axes[0, 0].scatter([0, 1], [2.0, 2.0])
        axes[0, 1].plot([0, 1], [100.0, 120.0])
        # The third axis is intentionally empty but shares with the first.
        harmonize_y_limits(axes, include_zero=True)
        self.assertEqual(axes[0, 0].get_ylim(), axes[0, 2].get_ylim())
        self.assertNotEqual(axes[0, 0].get_ylim(), axes[0, 1].get_ylim())
        self.assertLessEqual(axes[0, 0].get_ylim()[0], 0.0)
        plt.close(fig)

    def test_save_figure_writes_hashed_provenance_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "metrics.csv"
            source.write_text("metric,value\npearson,0.5\n", encoding="utf-8")
            fig, axes = comparison_subplots(1, 1)
            axes[0, 0].plot([0, 1], [0.2, 0.5])
            written = save_figure(
                fig,
                root / "figures" / "comparison",
                source_paths={"oof_metrics": source},
                metadata={"primary_metric": "pooled_five_fold_oof_pearson"},
                formats=("png", "svg"),
                close=True,
            )
            self.assertEqual(set(written), {"png", "svg", "provenance"})
            for path in written.values():
                self.assertTrue(path.is_file())
            payload = json.loads(written["provenance"].read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(
                payload["source_files"]["oof_metrics"]["sha256"],
                sha256_file(source),
            )
            self.assertEqual(
                payload["metadata"]["primary_metric"],
                "pooled_five_fold_oof_pearson",
            )
            self.assertEqual(
                payload["figure_files"]["png"]["sha256"],
                sha256_file(written["png"]),
            )


if __name__ == "__main__":
    unittest.main()
