# Lib1 Analysis And Figure Conventions

Use notebooks as a readable interpretation layer over tested analysis programs.
The notebook should not become a second implementation of OOF assembly, paired
RC comparisons, split membership, or specialized biological metrics.

For example, the July 2026 Stage 2 source of truth is
`src/analysis/lib1_dedup_stage2_analysis.py`. Run it first with
`--require-complete`, then let the notebook read its curated tables.

## Thin-Notebook Contract

Every new decision notebook should:

1. Find the repository with `find_repo_root()`; do not embed a
   machine-specific absolute path or assume the kernel's current directory.
2. Load a declared set of analysis outputs with `load_analysis_bundle()`.
3. Assert the expected campaign summary, required columns, unique keys,
   categorical levels, and paired comparison sides before ranking or plotting.
4. Use metrics produced by the tested analysis program. Do not silently
   redefine Pearson, coefficient-of-determination R2, OOF, RC pairing, or
   Intron sensitivity metrics in a notebook cell.
5. Declare which panels have comparable y axes. Panels may share a scale only
   when they show the same metric, units, transform, and estimand.
6. Save curated figures with their source-file hashes in a provenance sidecar.
7. Keep large executed notebooks and raw prediction tables in generated output
   roots. Commit only the source notebook and deliberately curated figures or
   compact scorecards.

The supporting API is `src/analysis/lib1_reporting.py`. It complements, rather
than replaces, `src/analysis/hpo_results_eval_utils.py`, which remains useful
for local W&B config/summary recovery and historical artifact-path
normalization.

## Loading A Contracted Analysis Bundle

```python
from src.analysis.lib1_reporting import (
    assert_exact_levels,
    assert_paired_keys,
    assert_unique_keys,
    find_repo_root,
    load_analysis_bundle,
    require_columns,
)

REPO = find_repo_root()
ANALYSIS_ROOT = REPO / "src/learn/outputs/analysis/lib1_dedup_stage2_july2026"

bundle = load_analysis_bundle(
    ANALYSIS_ROOT,
    required_files={
        "oof": "stage2_oof_metrics.csv",
        "rc": "stage2_rc_pair_metrics.csv",
        "rc_folds": "stage2_rc_fold_pair_metrics.csv",
        "intron": "stage2_intron_sensitivity_stratum_metrics.csv",
    },
    expected_summary={
        "analysis_cells": 660,
        "complete_oof_arms": 132,
        "complete_paired_rc_configs": 66,
        "primary_metric": "pooled_five_fold_oof_pearson",
        "audit_loader_instantiated": False,
        "require_complete": True,
    },
)

oof = bundle.table("oof")
require_columns(
    oof,
    ["analysis_lane", "part_slug", "base_config_id", "rc_mode",
     "pooled_oof_pearson"],
    "Stage 2 OOF metrics",
)
assert_unique_keys(
    oof,
    ["analysis_lane", "part_slug", "base_config_id", "rc_mode"],
    "Stage 2 OOF metrics",
)
assert_exact_levels(oof, "rc_mode", ["off", "on"], "Stage 2 OOF metrics")
assert_paired_keys(
    oof,
    ["analysis_lane", "part_slug", "base_config_id"],
    "rc_mode",
    table_name="Stage 2 OOF metrics",
)
```

Campaign-specific expected row counts and key definitions belong beside this
loading cell or in the tested campaign analyzer. They should not be hidden in a
generic plotting helper.

## Comparable Multi-Panel Figures

`comparison_subplots()` requires an explicit y-axis policy for every
multi-panel figure. This makes an accidental collection of independently
autoscaled panels fail early.

When every panel in a row has the same metric and estimand:

```python
from src.analysis.lib1_reporting import (
    comparison_subplots,
    harmonize_y_limits,
    save_figure,
)

fig, axes = comparison_subplots(2, 3, y_groups="row", figsize=(12, 7))
# Plot comparable values into each row...
harmonize_y_limits(axes, include_zero=True)
```

When columns, rather than rows, are the metric groups, use
`y_groups="column"`. For a custom layout, declare flattened panel indices or
`(row, column)` coordinates:

```python
fig, axes = comparison_subplots(
    2,
    3,
    y_groups=[[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]],
)
```

Use `y_groups="independent"` when panels intentionally have different units.
For example, loss, Pearson, and COD R2 epoch histories should not share one y
axis merely because they appear in the same row. Shared axes are a comparison
contract, not a cosmetic default. Keep that contract silent in presentation
titles: titles should state the scientific comparison and the unit represented
by each mark, not announce implementation details such as "shared y-axis" or
"identical limits." Record the scale policy in the figure provenance sidecar
and, only when an audience could reasonably misread it, in a caption.

After plotting, save both a notebook-friendly PNG and a publication-friendly
SVG, plus the source hashes:

```python
written = save_figure(
    fig,
    ANALYSIS_ROOT / "figures/stage2_oof_by_part",
    source_paths=bundle.provenance_sources(),
    metadata={
        "campaign_stage": bundle.summary["campaign_stage"],
        "primary_metric": bundle.summary["primary_metric"],
    },
)
written["provenance"]
```

## Reproducible Execution

The actual `boda_env` has `jupyter nbconvert`; it does not currently provide
Papermill or Jupytext. Execute a source notebook into a generated output root
instead of overwriting it:

```bash
cd "$(git rev-parse --show-toplevel)"
NOTEBOOK="tutorials/lib1_tasks/project/analysis.ipynb"
EXECUTED_DIR="src/learn/outputs/analysis/campaign/executed_notebooks"
conda run --no-capture-output -n boda_env \
  jupyter nbconvert \
  --to notebook \
  --execute "$NOTEBOOK" \
  --ExecutePreprocessor.timeout=1800 \
  --output-dir "$EXECUTED_DIR" \
  --output analysis__executed.ipynb
```

The notebook must remain independent of its execution directory by using
`find_repo_root()`. A failed contract assertion or cell execution is a failed
analysis build; do not publish partially executed output.

## Provenance And Output Policy

- Analysis programs write machine-readable tables under
  `src/learn/outputs/analysis/<campaign>/`.
- Source notebooks live under `tutorials/lib1_tasks/`.
- Curated paper-facing scorecards and figures may be copied to a clearly named
  notebook `plots/` or `presentation_plots/` directory when intentionally
  selected for version control.
- Raw predictions, per-epoch dumps, and executed notebooks stay generated and
  local unless a specific audit requires preservation.
- Every curated figure should retain the `.provenance.json` sidecar produced by
  `save_figure()` so its exact source tables can be identified later.

An optional Codex skill can eventually point agents to this convention and run
its checks. It should not duplicate any metric, path, or campaign logic from
the repository. Stabilize the code and this convention across at least two
analyses before encoding that thin workflow layer.
