import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.sequence_composition import (
    plot_positional_base_distribution,
    positional_base_distribution,
)


class PositionalBaseDistributionTests(unittest.TestCase):
    def test_grouped_counts_include_zero_cells_and_preserve_declared_order(self):
        result = positional_base_distribution(
            ["ac", "AT", "GC"],
            groups=["mask-1", "mask-1", "residual"],
        )

        self.assertEqual(result.shape, (2 * 2 * 4, 8))
        self.assertEqual(list(pd.unique(result["group"])), ["mask-1", "residual"])
        self.assertEqual(list(pd.unique(result["base"])), ["A", "C", "G", "T"])

        mask = result.loc[result["group"] == "mask-1"].set_index(
            ["position", "base"]
        )
        self.assertEqual(mask.loc[(1, "A"), "count"], 2)
        self.assertEqual(mask.loc[(1, "C"), "count"], 0)
        self.assertAlmostEqual(mask.loc[(2, "C"), "frequency"], 0.5)
        self.assertAlmostEqual(mask.loc[(2, "T"), "frequency"], 0.5)
        sums = result.groupby(["group", "position"])["frequency"].sum()
        np.testing.assert_allclose(sums.to_numpy(), 1.0)

    def test_weights_change_frequency_but_not_raw_count(self):
        result = positional_base_distribution(
            ["A", "C"], weights=[1.0, 3.0]
        ).set_index("base")

        self.assertEqual(result.loc["A", "count"], 1)
        self.assertEqual(result.loc["C", "count"], 1)
        self.assertAlmostEqual(result.loc["A", "weighted_count"], 1.0)
        self.assertAlmostEqual(result.loc["C", "weighted_count"], 3.0)
        self.assertAlmostEqual(result.loc["A", "frequency"], 0.25)
        self.assertAlmostEqual(result.loc["C", "frequency"], 0.75)
        self.assertTrue((result["total_weight"] == 4.0).all())

    def test_custom_alphabet_and_coordinate_start(self):
        result = positional_base_distribution(
            ["AN", "NN"], alphabet="ACGTN", position_start=0
        )
        self.assertEqual(result["position"].min(), 0)
        self.assertEqual(result["position"].max(), 1)
        self.assertIn("N", set(result["base"]))

    def test_invalid_sequence_and_weight_inputs_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            positional_base_distribution(["AC", "A"])
        with self.assertRaisesRegex(ValueError, "outside the declared alphabet"):
            positional_base_distribution(["AN"])
        with self.assertRaisesRegex(ValueError, "one label per sequence"):
            positional_base_distribution(["A", "C"], groups=["one"])
        with self.assertRaisesRegex(ValueError, "is missing"):
            positional_base_distribution(["A"], groups=[None])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            positional_base_distribution(["A"], weights=[-1])
        with self.assertRaisesRegex(ValueError, "zero total weight"):
            positional_base_distribution(["A"], weights=[0])


class PositionalBasePlotTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_frequency_heatmaps_use_absolute_common_scale(self):
        distribution = positional_base_distribution(
            ["AAC", "ATC", "GTC", "GTT"],
            groups=["mask-1", "mask-1", "residual", "residual"],
        )
        fig, axes = plot_positional_base_distribution(
            distribution,
            group_order=["residual", "mask-1"],
            title="Intron composition by inferred sensitivity category",
        )

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_title(loc="left"), "residual")
        self.assertEqual(axes[1].get_title(loc="left"), "mask-1")
        for axis in axes:
            self.assertEqual(axis.images[0].get_clim(), (0.0, 1.0))
            self.assertEqual(axis.images[0].get_array().shape, (4, 3))
        self.assertEqual(
            fig._suptitle.get_text(),
            "Intron composition by inferred sensitivity category",
        )

    def test_plot_rejects_incomplete_or_duplicate_grids(self):
        distribution = positional_base_distribution(["AC", "GT"])
        incomplete = distribution.iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "complete position x base grid"):
            plot_positional_base_distribution(incomplete)
        duplicate = pd.concat(
            [distribution, distribution.iloc[[0]]], ignore_index=True
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plot_positional_base_distribution(duplicate)

        invalid_sum = distribution.copy()
        invalid_sum.loc[0, "frequency"] = 0.25
        with self.assertRaisesRegex(ValueError, "sum to one"):
            plot_positional_base_distribution(invalid_sum)


if __name__ == "__main__":
    unittest.main()
