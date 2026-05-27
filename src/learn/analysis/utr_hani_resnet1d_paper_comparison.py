#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WORK_ROOT = Path("/home/minhang/synBio_AL")
REPO_ROOT = WORK_ROOT / "boda2_EU"
LEARN_ROOT = REPO_ROOT / "src" / "learn"
RUNS_CSV = LEARN_ROOT / "run_registry" / "runs.csv"
BEST_RUNS_CSV = LEARN_ROOT / "run_registry" / "best_runs.csv"
WANDB_ROOT = LEARN_ROOT / "wandb"
DATA_DIR = WORK_ROOT / "opt_EU_learn_n_design" / "utr_hani_2025" / "processed_utr_data"
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "tutorials"
    / "lib1_tasks"
    / "pretraining_CRE_public_data"
    / "presentation_plots"
    / "utr_hani_resnet1d_paper_comparison"
)

UTR_LIB1_WIDE = {
    "3UTR": DATA_DIR / "3UTR_lib1_branched_observed_heads.csv",
    "5UTR": DATA_DIR / "5UTR_lib1_branched_observed_heads.csv",
}
UTR_LIB2_PROCESSED = {
    "3UTR": DATA_DIR / "3UTR_lib2_processed.csv",
    "5UTR": DATA_DIR / "5UTR_lib2_processed.csv",
}

CELL_TYPE_NAMES = {
    "c1": "MDA-MB-231",
    "c2": "HepG2",
    "c4": "Jurkat",
    "c5": "BxPC-3",
    "c6": "SW480",
    "c13": "PA-1",
    "c15": "A549",
    "c17": "NALM6",
}
CELL_ORDER = ["c1", "c2", "c4", "c5", "c6", "c13", "c15", "c17"]
REGION_ORDER = ["3UTR", "5UTR"]
BIN_COLUMNS = ["1", "2", "3", "4"]

GROUPS = {
    "utr3__hani_rna_activity__observed_head_resnet1d_challenger": {
        "region": "3UTR",
        "variant": "ResNet1D broad HPO",
        "rank": 10,
        "default": True,
    },
    "utr3__hani_rna_activity__observed_head_resnet1d_focused_stage2": {
        "region": "3UTR",
        "variant": "ResNet1D focused HPO",
        "rank": 20,
        "default": True,
    },
    "utr5__hani_rna_activity__observed_head_resnet1d_challenger": {
        "region": "5UTR",
        "variant": "ResNet1D broad HPO",
        "rank": 10,
        "default": True,
    },
    "utr5__hani_rna_activity__observed_head_resnet1d_focused_stage2": {
        "region": "5UTR",
        "variant": "ResNet1D focused HPO",
        "rank": 20,
        "default": True,
    },
    "utr3__hani_rna_activity__observed_head_branched": {
        "region": "3UTR",
        "variant": "BassetBranched broad HPO",
        "rank": 30,
        "default": False,
    },
    "utr3__hani_rna_activity__observed_head_branched_focused_stage2": {
        "region": "3UTR",
        "variant": "BassetBranched focused HPO",
        "rank": 40,
        "default": False,
    },
    "utr5__hani_rna_activity__observed_head_branched": {
        "region": "5UTR",
        "variant": "BassetBranched broad HPO",
        "rank": 30,
        "default": False,
    },
    "utr5__hani_rna_activity__observed_head_branched_focused_stage2": {
        "region": "5UTR",
        "variant": "BassetBranched focused HPO",
        "rank": 40,
        "default": False,
    },
}

PARADE_SOURCES = {
    "paper_pubmed": "https://pubmed.ncbi.nlm.nih.gov/39803435/",
    "paper_doi": "https://doi.org/10.1101/2024.12.31.630783",
    "3UTR_lib1": "https://github.com/autosome-ru/parade/blob/master/benchmark/model-collection/utr3_eval.ipynb",
    "5UTR_lib1": "https://github.com/autosome-ru/parade/blob/master/benchmark/model-collection/utr5_eval.ipynb",
    "3UTR_lib2": "https://github.com/autosome-ru/parade/blob/master/benchmark/model-collection/utr3_eval_lib2.ipynb",
    "5UTR_lib2": "https://github.com/autosome-ru/parade/blob/master/benchmark/model-collection/utr5_eval_lib2.ipynb",
}

PARADE_LEGNET_VALUES = {
    ("3UTR", "lib1_test"): {
        "c1": 0.737771,
        "c13": 0.654717,
        "c17": 0.719844,
        "c2": 0.750149,
        "c4": 0.650344,
        "c6": 0.728190,
        "mean": 0.792601,
        "source_key": "3UTR_lib1",
    },
    ("5UTR", "lib1_test"): {
        "c1": 0.774258,
        "c17": 0.741030,
        "c2": 0.788963,
        "c4": 0.647458,
        "c6": 0.781765,
        "mean": 0.823008,
        "source_key": "5UTR_lib1",
    },
    ("3UTR", "lib2_overlap"): {
        "c1": 0.669872,
        "c17": 0.649836,
        "c2": 0.681765,
        "c4": 0.674730,
        "c6": 0.696651,
        "mean": 0.767333,
        "source_key": "3UTR_lib2",
    },
    ("5UTR", "lib2_overlap"): {
        "c1": 0.571283,
        "c17": 0.623088,
        "c2": 0.523317,
        "c4": 0.584193,
        "c6": 0.388581,
        "mean": 0.704422,
        "source_key": "5UTR_lib2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Hani UTR ResNet1D vs PARADE per-cell and Lib2 comparison plots."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--work-root", type=Path, default=WORK_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--include-basset-context",
        action="store_true",
        help="Also include validation winners from BassetBranched broad/focused groups.",
    )
    parser.add_argument(
        "--skip-artifact-eval",
        action="store_true",
        help="Only use local W&B summaries; skip Lib1/Lib2 artifact prediction.",
    )
    parser.add_argument(
        "--extra-run-id",
        action="append",
        default=[],
        help="Additional run_id to include if present in runs.csv. May be repeated.",
    )
    parser.add_argument(
        "--max-artifacts-per-region",
        type=int,
        default=2,
        help=(
            "Limit artifact Lib2 evaluation to this many selected BODA runs per region. "
            "The best validation metric runs are kept first."
        ),
    )
    return parser.parse_args()


def update_roots(args: argparse.Namespace) -> None:
    global REPO_ROOT, WORK_ROOT, LEARN_ROOT, RUNS_CSV, BEST_RUNS_CSV, WANDB_ROOT, DATA_DIR
    global DEFAULT_OUTDIR, UTR_LIB1_WIDE, UTR_LIB2_PROCESSED

    REPO_ROOT = args.repo_root
    WORK_ROOT = args.work_root
    LEARN_ROOT = REPO_ROOT / "src" / "learn"
    RUNS_CSV = LEARN_ROOT / "run_registry" / "runs.csv"
    BEST_RUNS_CSV = LEARN_ROOT / "run_registry" / "best_runs.csv"
    WANDB_ROOT = LEARN_ROOT / "wandb"
    DATA_DIR = WORK_ROOT / "opt_EU_learn_n_design" / "utr_hani_2025" / "processed_utr_data"
    UTR_LIB1_WIDE = {
        "3UTR": DATA_DIR / "3UTR_lib1_branched_observed_heads.csv",
        "5UTR": DATA_DIR / "5UTR_lib1_branched_observed_heads.csv",
    }
    UTR_LIB2_PROCESSED = {
        "3UTR": DATA_DIR / "3UTR_lib2_processed.csv",
        "5UTR": DATA_DIR / "5UTR_lib2_processed.csv",
    }


def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
        return value.split()
    return [value]


def cell_label(head: str) -> str:
    name = CELL_TYPE_NAMES.get(head)
    return f"{head} {name}" if name else head


def cell_sort_key(head: str) -> int:
    try:
        return CELL_ORDER.index(head)
    except ValueError:
        return len(CELL_ORDER) + 1


def read_runs(include_basset_context: bool) -> pd.DataFrame:
    runs = pd.read_csv(RUNS_CSV)
    numeric_cols = [
        "best_metric_value",
        "val_r2",
        "val_pearson",
        "test_r2",
        "test_pearson",
        "train_r2",
        "train_pearson",
    ]
    runs = coerce_numeric(runs, numeric_cols)
    active_groups = {
        group: meta
        for group, meta in GROUPS.items()
        if meta["default"] or include_basset_context
    }
    runs = runs[runs["comparison_group"].isin(active_groups)].copy()
    if "status" in runs.columns:
        runs = runs[runs["status"].fillna("completed").eq("completed")].copy()
    runs["region"] = runs["comparison_group"].map(lambda g: active_groups[g]["region"])
    runs["variant"] = runs["comparison_group"].map(lambda g: active_groups[g]["variant"])
    runs["variant_rank"] = runs["comparison_group"].map(lambda g: active_groups[g]["rank"])
    runs["selection_source"] = "validation_winner_by_group"
    return runs


def select_validation_winners(runs: pd.DataFrame, extra_run_ids: Sequence[str]) -> pd.DataFrame:
    if runs.empty:
        return runs.copy()

    winners = (
        runs.sort_values("best_metric_value", ascending=False)
        .groupby(["region", "comparison_group"], as_index=False, observed=True)
        .head(1)
        .copy()
    )

    extras = runs[runs["run_id"].astype(str).isin([str(x) for x in extra_run_ids])].copy()
    if not extras.empty:
        extras["selection_source"] = "extra_run_id"
        winners = pd.concat([winners, extras], ignore_index=True)

    if BEST_RUNS_CSV.exists():
        best = pd.read_csv(BEST_RUNS_CSV)
        best = best[
            best["registry_status"].eq("current")
            & best["task_family"].isin(["utr3", "utr5"])
            & best["target_family"].eq("hani_rna_activity")
        ].copy()
        best_ids = set(best["run_id"].dropna().astype(str))
        best_rows = runs[runs["run_id"].astype(str).isin(best_ids)].copy()
        if not best_rows.empty:
            best_rows["selection_source"] = "best_runs_current"
            winners = pd.concat([winners, best_rows], ignore_index=True)

    winners = winners.sort_values(
        ["region", "variant_rank", "best_metric_value"],
        ascending=[True, True, False],
    ).drop_duplicates("run_id", keep="first")
    winners["model_label"] = (
        winners["region"].astype(str)
        + " "
        + winners["variant"].astype(str)
        + " "
        + winners["run_id"].astype(str)
    )
    return winners.reset_index(drop=True)


def load_local_summary(run_id: str) -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    matches = sorted(WANDB_ROOT.glob(f"run-*{run_id}/files/wandb-summary.json"))
    if not matches:
        return None, None
    return matches[0], json.loads(matches[0].read_text())


def extract_per_head_rows(run_row: pd.Series) -> List[Dict[str, Any]]:
    summary_path, summary = load_local_summary(str(run_row["run_id"]))
    rows: List[Dict[str, Any]] = []
    if summary is None:
        return rows
    pattern = re.compile(r"^(val|test|train)_(pearson|spearman|mse|pearson_squared)_(.+)$")
    for key, value in summary.items():
        match = pattern.match(key)
        if not match:
            continue
        split, metric, head = match.groups()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        rows.append(
            {
                "run_id": run_row["run_id"],
                "region": run_row["region"],
                "variant": run_row["variant"],
                "model_module": run_row.get("model_module", ""),
                "model_label": run_row["model_label"],
                "split": split,
                "metric": metric,
                "head": head,
                "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
                "cell_label": cell_label(head),
                "value": float(value),
                "summary_path": str(summary_path),
            }
        )
    return rows


def build_parade_reference() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (region, split), values in PARADE_LEGNET_VALUES.items():
        source_key = str(values["source_key"])
        per_cell_values = {
            head: float(value)
            for head, value in values.items()
            if head not in {"source_key", "mean"}
        }
        for head, value in per_cell_values.items():
            rows.append(
                {
                    "region": region,
                    "split": split,
                    "model_label": "PARADE LegNetClassifier",
                    "source": "PARADE authors public notebook",
                    "metric": "pearson",
                    "head": head,
                    "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
                    "cell_label": cell_label(head),
                    "value": value,
                    "source_url": PARADE_SOURCES[source_key],
                }
            )
        rows.append(
            {
                "region": region,
                "split": split,
                "model_label": "PARADE LegNetClassifier",
                "source": "PARADE authors public notebook",
                "metric": "average_activity_pearson",
                "head": "mean",
                "cell_type_name": "mean",
                "cell_label": "mean",
                "value": float(values["mean"]),
                "source_url": PARADE_SOURCES[source_key],
            }
        )
        rows.append(
            {
                "region": region,
                "split": split,
                "model_label": "PARADE LegNetClassifier",
                "source": "PARADE authors public notebook",
                "metric": "mean_per_cell_pearson",
                "head": "mean_per_cell",
                "cell_type_name": "mean per cell",
                "cell_label": "mean per cell",
                "value": float(np.mean(list(per_cell_values.values()))),
                "source_url": PARADE_SOURCES[source_key],
            }
        )
    return pd.DataFrame(rows)


def grouped_barplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    ax: plt.Axes,
    title: str,
    y_label: str = "Pearson r",
    y_lim: Tuple[float, float] = (0.0, 0.9),
) -> None:
    if df.empty:
        ax.set_axis_off()
        ax.set_title(title)
        return

    x_values = list(dict.fromkeys(df[x_col].tolist()))
    hue_values = list(dict.fromkeys(df[hue_col].tolist()))
    x = np.arange(len(x_values))
    width = min(0.82 / max(len(hue_values), 1), 0.28)
    offsets = (np.arange(len(hue_values)) - (len(hue_values) - 1) / 2.0) * width

    cmap = plt.get_cmap("tab10")
    for idx, hue in enumerate(hue_values):
        sub = df[df[hue_col].eq(hue)]
        values = []
        for x_value in x_values:
            series = sub[sub[x_col].eq(x_value)][y_col]
            values.append(float(series.iloc[0]) if len(series) else np.nan)
        ax.bar(x + offsets[idx], values, width=width, label=hue, color=cmap(idx % 10))

    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=35, ha="right")
    ax.set_ylim(*y_lim)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def plot_per_cell_lib1(per_head: pd.DataFrame, parade: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    ours = per_head[
        per_head["split"].eq("test")
        & per_head["metric"].eq("pearson")
        & per_head["head"].isin(CELL_TYPE_NAMES)
    ].copy()
    ours["source"] = "BODA local W&B summary"
    ours = ours.rename(columns={"value": "pearson"})

    paper = parade[
        parade["split"].eq("lib1_test")
        & parade["metric"].eq("pearson")
        & parade["head"].isin(CELL_TYPE_NAMES)
    ].copy()
    paper = paper.rename(columns={"value": "pearson"})

    cols = [
        "region",
        "model_label",
        "source",
        "head",
        "cell_type_name",
        "cell_label",
        "pearson",
    ]
    combined = pd.concat([ours[cols], paper[cols]], ignore_index=True)
    combined["head_rank"] = combined["head"].map(cell_sort_key)
    combined = combined.sort_values(["region", "head_rank", "model_label"])
    combined.to_csv(outdir / "utr_hani_per_cell_lib1_test_pearson_vs_parade.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, region in zip(axes, REGION_ORDER):
        region_df = combined[combined["region"].eq(region)].copy()
        region_df = region_df.sort_values(["head_rank", "model_label"])
        grouped_barplot(
            region_df,
            x_col="cell_label",
            y_col="pearson",
            hue_col="model_label",
            ax=ax,
            title=f"{region} Lib1 held-out test per-cell Pearson",
            y_lim=(0.0, 0.9),
        )
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].legend_.remove() if axes[0].legend_ else None
    fig.tight_layout()
    fig.savefig(outdir / "utr_hani_per_cell_lib1_test_pearson_vs_parade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    matrix = combined.pivot_table(
        index=["region", "model_label"],
        columns="cell_label",
        values="pearson",
        aggfunc="first",
    )
    matrix = matrix.reindex(
        columns=[cell_label(h) for h in CELL_ORDER if cell_label(h) in matrix.columns]
    )
    matrix.to_csv(outdir / "utr_hani_per_cell_lib1_test_pearson_heatmap_matrix.csv")
    if not matrix.empty:
        fig_h = max(3.2, 0.42 * len(matrix.index) + 1.2)
        fig_w = max(8.0, 0.92 * len(matrix.columns) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, aspect="auto", vmin=0.0, vmax=0.9, cmap="viridis")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_yticklabels([f"{r} {m}" for r, m in matrix.index])
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if np.isfinite(values[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if values[i, j] < 0.62 else "black",
                        fontsize=8,
                    )
        ax.set_title("Lib1 held-out test Pearson by cell type")
        fig.colorbar(image, ax=ax, label="Pearson r")
        fig.tight_layout()
        fig.savefig(outdir / "utr_hani_per_cell_lib1_test_pearson_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return combined


def pearson_np(a: Any, b: Any) -> float:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 2:
        return np.nan
    corr = np.corrcoef(a_arr[mask], b_arr[mask])[0, 1]
    return float(corr) if np.isfinite(corr) else np.nan


def aggregate_processed_library(path: Path, heads: Sequence[str], require_all: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in BIN_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = df[df["cell_type"].isin(heads)].dropna(subset=["seq", "cell_type"] + BIN_COLUMNS).copy()
    group_cols = ["seq", "cell_type"] + (["fold"] if "fold" in keep.columns else [])
    grouped = keep.groupby(group_cols, observed=True)[BIN_COLUMNS].sum().reset_index()
    denom = grouped[BIN_COLUMNS].sum(axis=1)
    grouped = grouped[denom > 0].copy()
    denom = denom[denom > 0]
    grouped["rna_activity"] = (
        grouped["1"] + 2 * grouped["2"] + 3 * grouped["3"] + 4 * grouped["4"]
    ) / denom
    index_cols = ["seq"] + (["fold"] if "fold" in grouped.columns else [])
    wide = grouped.pivot_table(
        index=index_cols,
        columns="cell_type",
        values="rna_activity",
        aggfunc="first",
    ).reset_index()
    for head in heads:
        if head not in wide.columns:
            wide[head] = np.nan
    if require_all:
        wide = wide.dropna(subset=list(heads)).copy()
    return wide[["seq"] + (["fold"] if "fold" in wide.columns else []) + list(heads)]


def safe_extract(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([str(target_root), str(member_path)]) != str(target_root):
            raise RuntimeError(f"Unsafe path in tarball: {member.name}")
    tar.extractall(str(target_dir))


def import_boda_stack() -> Tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import boda
    from boda.common import utils as boda_utils

    return torch, DataLoader, TensorDataset, (boda, boda_utils)


def load_artifact_model(artifact_path: Path, torch: Any, boda: Any) -> Tuple[Any, Dict[str, Any], Dict[str, Any], str]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(str(artifact_path)) as tar:
            safe_extract(tar, tmp_path)
        ckpt = torch.load(tmp_path / "artifacts" / "torch_checkpoint.pt", map_location="cpu")
    model_module = ckpt["model_module"]
    model_cls = getattr(boda.model, model_module)
    model_hparams = vars(ckpt["model_hparams"])
    model = model_cls(**model_hparams)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vars(ckpt["data_hparams"]), model_hparams, str(model_module)


def predict_sequences(
    model: Any,
    seqs: Sequence[str],
    batch_size: int,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    boda_utils: Any,
) -> np.ndarray:
    x = torch.stack([boda_utils.dna2tensor(str(seq).upper()) for seq in seqs])
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (batch_x,) in loader:
            preds.append(model(batch_x).detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def evaluate_artifact(
    run_row: pd.Series,
    torch: Any,
    DataLoader: Any,
    TensorDataset: Any,
    boda: Any,
    boda_utils: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    region = str(run_row["region"])
    artifact_path = Path(str(run_row["artifact_path"]))
    model, data_hparams, model_hparams, model_module = load_artifact_model(artifact_path, torch, boda)
    heads = [str(x) for x in as_list(data_hparams["activity_columns"])]
    lib1 = pd.read_csv(UTR_LIB1_WIDE[region])
    train = lib1[lib1["fold"].eq("train")]
    means = train[heads].mean()
    stds = train[heads].std().replace(0, 1.0)

    eval_sets = [
        ("lib1_test", lib1[lib1["fold"].eq("test")].copy(), heads),
        ("lib2_overlap", aggregate_processed_library(UTR_LIB2_PROCESSED[region], heads, require_all=False), None),
    ]

    rows: List[Dict[str, Any]] = []
    per_head: List[Dict[str, Any]] = []
    batch_size = int(data_hparams.get("batch_size", 512))

    for split, df, available_heads in eval_sets:
        if available_heads is None:
            available_heads = [
                head
                for head in heads
                if head in df.columns and pd.to_numeric(df[head], errors="coerce").notna().sum() >= 2
            ]
            df = df.dropna(subset=available_heads).copy()
        if df.empty or not available_heads:
            continue

        pred_norm = predict_sequences(
            model,
            df["seq"].tolist(),
            batch_size=batch_size,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            boda_utils=boda_utils,
        )
        pred_norm_df = pd.DataFrame(pred_norm, columns=heads, index=df.index)
        pred_raw = pred_norm_df.mul(stds, axis=1).add(means, axis=1)
        true_raw = df[heads].astype(float)

        pearsons = []
        for head in available_heads:
            r = pearson_np(pred_raw[head], true_raw[head])
            pearsons.append(r)
            per_head.append(
                {
                    "run_id": run_row["run_id"],
                    "region": region,
                    "variant": run_row["variant"],
                    "model_module": model_module,
                    "model_label": run_row["model_label"],
                    "split": split,
                    "head": head,
                    "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
                    "cell_label": cell_label(head),
                    "pearson": r,
                    "pearson_r2": r ** 2 if np.isfinite(r) else np.nan,
                    "n_sequences": len(df),
                }
            )

        flat_r = pearson_np(
            pred_raw[available_heads].to_numpy().ravel(),
            true_raw[available_heads].to_numpy().ravel(),
        )
        avg_r = pearson_np(
            pred_raw[available_heads].mean(axis=1),
            true_raw[available_heads].mean(axis=1),
        )
        mean_per_head = float(np.nanmean(pearsons)) if len(pearsons) else np.nan
        rows.append(
            {
                "run_id": run_row["run_id"],
                "region": region,
                "variant": run_row["variant"],
                "model_module": model_module,
                "model_label": run_row["model_label"],
                "split": split,
                "n_sequences": len(df),
                "heads_used": ",".join(available_heads),
                "n_heads": len(available_heads),
                "mean_per_head_pearson": mean_per_head,
                "mean_per_head_pearson_r2": mean_per_head ** 2 if np.isfinite(mean_per_head) else np.nan,
                "flattened_raw_pearson": flat_r,
                "flattened_raw_pearson_r2": flat_r ** 2 if np.isfinite(flat_r) else np.nan,
                "average_activity_pearson": avg_r,
                "average_activity_pearson_r2": avg_r ** 2 if np.isfinite(avg_r) else np.nan,
                "model_params": int(sum(p.numel() for p in model.parameters())),
                "artifact_path": str(artifact_path),
                "model_hparams": json.dumps(model_hparams, sort_keys=True),
            }
        )
    return rows, per_head


def select_artifacts_for_eval(selected: pd.DataFrame, max_per_region: int) -> pd.DataFrame:
    candidates = selected[
        selected["artifact_path"].notna() & selected["artifact_path"].astype(str).ne("")
    ].copy()
    if candidates.empty:
        return candidates
    candidates["artifact_exists"] = candidates["artifact_path"].map(lambda p: Path(str(p)).exists())
    candidates = candidates[candidates["artifact_exists"]].copy()
    candidates = candidates.sort_values(
        ["region", "best_metric_value"],
        ascending=[True, False],
    )
    if max_per_region > 0:
        candidates = candidates.groupby("region", as_index=False, observed=True).head(max_per_region)
    return candidates.reset_index(drop=True)


def run_artifact_eval(selected: pd.DataFrame, max_per_region: int, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = select_artifacts_for_eval(selected, max_per_region)
    artifacts.to_csv(outdir / "utr_hani_artifact_eval_selected_runs.csv", index=False)
    if artifacts.empty:
        print("No selected local artifact tarballs found; skipping Lib1/Lib2 prediction eval.")
        return pd.DataFrame(), pd.DataFrame()

    try:
        torch, DataLoader, TensorDataset, boda_tuple = import_boda_stack()
        boda, boda_utils = boda_tuple
    except Exception as exc:
        print("Torch/BODA imports failed; skipping Lib1/Lib2 artifact prediction eval.")
        print(repr(exc))
        return pd.DataFrame(), pd.DataFrame()

    eval_rows: List[Dict[str, Any]] = []
    per_head_rows: List[Dict[str, Any]] = []
    for row in artifacts.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        print(f"Evaluating {row_s['region']} {row_s['variant']} {row_s['run_id']} on Lib1 test and Lib2 overlap")
        rows, head_rows = evaluate_artifact(
            row_s,
            torch=torch,
            DataLoader=DataLoader,
            TensorDataset=TensorDataset,
            boda=boda,
            boda_utils=boda_utils,
        )
        eval_rows.extend(rows)
        per_head_rows.extend(head_rows)

    eval_df = pd.DataFrame(eval_rows)
    per_head_df = pd.DataFrame(per_head_rows)
    if not eval_df.empty:
        eval_df.to_csv(outdir / "utr_hani_lib1_lib2_artifact_eval_summary.csv", index=False)
    if not per_head_df.empty:
        per_head_df.to_csv(outdir / "utr_hani_lib1_lib2_artifact_eval_per_cell.csv", index=False)
    return eval_df, per_head_df


def plot_average_activity(eval_df: pd.DataFrame, parade: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not eval_df.empty:
        for rec in eval_df.to_dict("records"):
            rows.append(
                {
                    "region": rec["region"],
                    "split": rec["split"],
                    "model_label": rec["model_label"],
                    "source": "BODA artifact prediction",
                    "average_activity_pearson": rec["average_activity_pearson"],
                    "mean_per_head_pearson": rec["mean_per_head_pearson"],
                    "n_sequences": rec["n_sequences"],
                    "heads_used": rec["heads_used"],
                }
            )

    for (region, split), sub in parade.groupby(["region", "split"], observed=True):
        avg = sub[sub["metric"].eq("average_activity_pearson")]["value"]
        mean_cell = sub[sub["metric"].eq("mean_per_cell_pearson")]["value"]
        rows.append(
            {
                "region": region,
                "split": split,
                "model_label": "PARADE LegNetClassifier",
                "source": "PARADE authors public notebook",
                "average_activity_pearson": float(avg.iloc[0]) if len(avg) else np.nan,
                "mean_per_head_pearson": float(mean_cell.iloc[0]) if len(mean_cell) else np.nan,
                "n_sequences": np.nan,
                "heads_used": ",".join(sorted(sub[sub["metric"].eq("pearson")]["head"], key=cell_sort_key)),
            }
        )

    combined = pd.DataFrame(rows)
    if combined.empty:
        return combined
    combined["split_label"] = combined["region"] + " " + combined["split"].map(
        {"lib1_test": "Lib1 test", "lib2_overlap": "Lib2 overlap"}
    ).fillna(combined["split"])
    split_order = [
        "3UTR Lib1 test",
        "3UTR Lib2 overlap",
        "5UTR Lib1 test",
        "5UTR Lib2 overlap",
    ]
    combined["split_rank"] = combined["split_label"].map(
        {name: idx for idx, name in enumerate(split_order)}
    ).fillna(99)
    combined = combined.sort_values(["split_rank", "model_label"])
    combined.to_csv(outdir / "utr_hani_average_activity_vs_parade.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.5), sharey=True)
    for ax, metric, title in zip(
        axes,
        ["average_activity_pearson", "mean_per_head_pearson"],
        ["Paper-style average activity Pearson", "Mean per-cell Pearson"],
    ):
        grouped_barplot(
            combined,
            x_col="split_label",
            y_col=metric,
            hue_col="model_label",
            ax=ax,
            title=title,
            y_lim=(0.0, 0.9),
        )
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].legend_.remove() if axes[0].legend_ else None
    fig.tight_layout()
    fig.savefig(outdir / "utr_hani_average_activity_vs_parade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return combined


def plot_lib2_per_cell(eval_per_head: pd.DataFrame, parade: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    if not eval_per_head.empty:
        ours = eval_per_head[
            eval_per_head["split"].eq("lib2_overlap") & eval_per_head["head"].isin(CELL_TYPE_NAMES)
        ].copy()
        ours["source"] = "BODA artifact prediction"
        ours = ours.rename(columns={"pearson": "value"})
        rows.append(ours[["region", "model_label", "source", "head", "cell_type_name", "cell_label", "value"]])

    paper = parade[
        parade["split"].eq("lib2_overlap")
        & parade["metric"].eq("pearson")
        & parade["head"].isin(CELL_TYPE_NAMES)
    ].copy()
    if not paper.empty:
        rows.append(paper[["region", "model_label", "source", "head", "cell_type_name", "cell_label", "value"]])

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined["head_rank"] = combined["head"].map(cell_sort_key)
    combined = combined.sort_values(["region", "head_rank", "model_label"])
    combined.to_csv(outdir / "utr_hani_per_cell_lib2_overlap_pearson_vs_parade.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, region in zip(axes, REGION_ORDER):
        region_df = combined[combined["region"].eq(region)].copy()
        grouped_barplot(
            region_df,
            x_col="cell_label",
            y_col="value",
            hue_col="model_label",
            ax=ax,
            title=f"{region} Lib2 overlap per-cell Pearson",
            y_lim=(0.0, 0.9),
        )
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].legend_.remove() if axes[0].legend_ else None
    fig.tight_layout()
    fig.savefig(outdir / "utr_hani_per_cell_lib2_overlap_pearson_vs_parade.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return combined


def write_metadata(outdir: Path, selected: pd.DataFrame) -> None:
    metadata = {
        "description": "Hani UTR ResNet1D focused-HPO comparison against PARADE LegNetClassifier public notebook metrics.",
        "cell_type_mapping": CELL_TYPE_NAMES,
        "parade_sources": PARADE_SOURCES,
        "selected_run_ids": selected["run_id"].astype(str).tolist() if not selected.empty else [],
        "selection_rule": "validation winner per included comparison group; current best_runs rows and --extra-run-id rows are retained if present",
    }
    (outdir / "utr_hani_resnet1d_paper_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    update_roots(args)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    runs = read_runs(include_basset_context=args.include_basset_context)
    selected = select_validation_winners(runs, args.extra_run_id)
    selected.to_csv(outdir / "utr_hani_selected_validation_winners_for_paper_comparison.csv", index=False)
    write_metadata(outdir, selected)

    parade = build_parade_reference()
    parade.to_csv(outdir / "utr_hani_parade_legnetclassifier_reference_values.csv", index=False)

    per_head = pd.DataFrame(
        item
        for _, run_row in selected.iterrows()
        for item in extract_per_head_rows(run_row)
    )
    if per_head.empty:
        print("No local per-head W&B summaries found for selected BODA runs.")
    else:
        per_head.to_csv(outdir / "utr_hani_selected_run_per_head_wandb_metrics.csv", index=False)
        plot_per_cell_lib1(per_head, parade, outdir)

    if args.skip_artifact_eval:
        print("--skip-artifact-eval set; not computing Lib1/Lib2 artifact predictions.")
        eval_df = pd.DataFrame()
        eval_per_head = pd.DataFrame()
    else:
        eval_df, eval_per_head = run_artifact_eval(
            selected,
            max_per_region=args.max_artifacts_per_region,
            outdir=outdir,
        )

    plot_average_activity(eval_df, parade, outdir)
    plot_lib2_per_cell(eval_per_head, parade, outdir)

    print(f"Wrote comparison outputs to: {outdir}")
    if not selected.empty:
        print(
            selected[
                [
                    "region",
                    "variant",
                    "run_id",
                    "best_metric_value",
                    "test_pearson",
                    "artifact_path",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
