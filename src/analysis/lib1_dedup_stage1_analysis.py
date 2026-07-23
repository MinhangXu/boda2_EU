"""Reproducible analysis helpers for the July 2026 Lib1 dedup Stage 1 replay."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARN_ROOT = REPO_ROOT / "src" / "learn"
CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
EXACT_STAGE = "stage1_exact_replay"
CALIBRATION_STAGE = "stage1_pre_dedup_calibration"
DEFAULT_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_phase1_exact_replay_july2026__run_manifest.jsonl"
)
DEFAULT_REGISTRY = LEARN_ROOT / "run_registry/runs.csv"

INTRON_MASKS = {
    "mask1_specific": "GTRHKHNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNYHYNYYYYYYYYYYYYYYYYYNYAG",
    "mask2": "GTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAG",
    "mask3": "NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN",
}
IUPAC_REGEX = {
    "A": "A", "C": "C", "G": "G", "T": "T", "R": "[AG]",
    "Y": "[CT]", "H": "[ACT]", "K": "[GT]", "N": "[ACGT]",
}

NUMERIC_COLUMNS = [
    "best_epoch",
    "best_metric_value",
    "val_loss",
    "val_pearson",
    "val_spearman",
    "val_cod_r2",
    "val_mse",
    "train_loss",
    "train_pearson",
    "train_spearman",
    "train_cod_r2",
    "train_mse",
]


def _json_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _jsonable(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def load_stage1_results(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> pd.DataFrame:
    """Join the immutable manifest to the append-only registry and flatten configs."""
    manifest_rows = _json_lines(Path(manifest_path))
    flat_manifest = []
    for row in manifest_rows:
        record = {
            "manifest_row": int(row["manifest_row"]),
            "run_kind": row["run_kind"],
            "planned_run_name": row["planned_run_name"],
            "part_slug": row["part_slug"],
            "lane_id": row["lane_id"],
            "manifest_architecture": row["architecture"],
            "base_config_id_manifest": row["base_config_id"],
            "base_config_sha256": row["base_config_sha256"],
            "row_fingerprint": row["row_fingerprint"],
            "dataset_path": row["dataset_path"],
            "split_manifest_path": row["split_manifest_path"],
            "source_kind": ";".join(
                sorted({source["candidate_kind"] for source in row["source_candidates"]})
            ),
            "historical_validation_mean": row.get("historical_validation_mean"),
        }
        record.update(
            {f"hp_{key}": _jsonable(value) for key, value in row["base_identity"].items()}
        )
        flat_manifest.append(record)
    manifest = pd.DataFrame(flat_manifest)

    with Path(registry_path).open(newline="", encoding="utf-8") as handle:
        registry = pd.DataFrame(list(csv.DictReader(handle)))
    registry = registry.loc[registry["campaign_id"].eq(CAMPAIGN_ID)].copy()
    registry = registry.loc[
        registry["campaign_stage"].isin([EXACT_STAGE, CALIBRATION_STAGE])
    ].copy()
    registry = registry.sort_values("timestamp").drop_duplicates("run_name", keep="last")
    for column in NUMERIC_COLUMNS:
        registry[column] = pd.to_numeric(registry[column], errors="coerce")

    merged = manifest.merge(
        registry,
        left_on="planned_run_name",
        right_on="run_name",
        how="left",
        validate="one_to_one",
        suffixes=("_manifest", ""),
    )
    # ``part_slug`` became a registry column when the Stage 2 schema was
    # appended. Historical Stage 1 rows legitimately leave that newer column
    # blank, so the immutable Stage 1 manifest remains authoritative here.
    if "part_slug_manifest" in merged:
        merged["part_slug"] = merged["part_slug"].where(
            merged["part_slug"].astype(str).ne(""),
            merged["part_slug_manifest"],
        )
    merged["val_rmse"] = np.sqrt(merged["val_mse"])
    merged["train_rmse"] = np.sqrt(merged["train_mse"])
    merged["train_minus_val_pearson"] = merged["train_pearson"] - merged["val_pearson"]
    merged["prediction_exists"] = merged["prediction_path"].map(
        lambda value: bool(value) and Path(value).is_file()
    )
    merged["provenance_path"] = merged.apply(
        lambda row: str(
            Path(row["prediction_path"]).parents[1]
            / "provenance"
            / f"{row['run_id']}__run_provenance.json"
        )
        if row.get("prediction_path")
        else "",
        axis=1,
    )
    merged["provenance_exists"] = merged["provenance_path"].map(
        lambda value: bool(value) and Path(value).is_file()
    )
    return merged.sort_values("manifest_row").reset_index(drop=True)


def completion_audit(results: pd.DataFrame) -> pd.DataFrame:
    """Return compact contract checks; every row should report ``True``."""
    exact = results.loc[results["run_kind"].eq("exact_replay")]
    calibration = results.loc[results["run_kind"].eq("pre_dedup_calibration")]
    test_columns = [
        column
        for column in results.columns
        if column.startswith("test_") and column not in {"test_min_barcodes"}
    ]
    checks = {
        "manifest_rows_910": len(results) == 910,
        "exact_rows_885": len(exact) == 885,
        "calibration_rows_25": len(calibration) == 25,
        "registry_rows_present": results["run_id"].notna().all(),
        "all_completed": results["status"].eq("completed").all(),
        "unique_run_names": results["run_name"].nunique() == 910,
        "unique_row_fingerprints": results["row_fingerprint"].nunique() == 910,
        "all_predictions_present": results["prediction_exists"].all(),
        "all_provenance_present": results["provenance_exists"].all(),
        "no_test_metrics": not test_columns or results[test_columns].replace("", np.nan).isna().all().all(),
        "exact_fold0_only": exact["development_fold"].astype(str).eq("0").all(),
        "exact_seed1701_only": exact["model_seed"].astype(str).eq("1701").all(),
        "exact_unweighted_only": exact["loss_mode"].eq("unweighted_mse").all(),
    }
    return pd.DataFrame({"check": checks.keys(), "passed": checks.values()})


def load_predictions(row: pd.Series | dict) -> pd.DataFrame:
    frame = pd.read_csv(row["prediction_path"], sep="\t")
    required = {
        "construct_id",
        "n_barcodes",
        "target_processed",
        "prediction_processed",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
    return frame


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    target = frame["target_processed"].to_numpy(float)
    prediction = frame["prediction_processed"].to_numpy(float)
    mse = float(np.mean((target - prediction) ** 2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    cod_r2 = float(1.0 - np.sum((target - prediction) ** 2) / denominator)
    return {
        "prediction_pearson": float(pearsonr(target, prediction)[0]),
        "prediction_spearman": float(spearmanr(target, prediction)[0]),
        "prediction_mse": mse,
        "prediction_rmse": math.sqrt(mse),
        "prediction_cod_r2": cod_r2,
    }


def verify_prediction_metrics(
    results: pd.DataFrame,
    sample: int | None = None,
    random_state: int = 1701,
) -> pd.DataFrame:
    """Recompute final metrics from exported predictions for all or a sample."""
    use = results if sample is None or sample >= len(results) else results.sample(sample, random_state=random_state)
    records = []
    for _, row in use.iterrows():
        metrics = prediction_metrics(load_predictions(row))
        records.append(
            {
                "manifest_row": row["manifest_row"],
                "run_id": row["run_id"],
                **metrics,
                "pearson_abs_error": abs(metrics["prediction_pearson"] - row["val_pearson"]),
                "mse_abs_error": abs(metrics["prediction_mse"] - row["val_mse"]),
            }
        )
    return pd.DataFrame(records)


def calibration_pairs(results: pd.DataFrame) -> pd.DataFrame:
    """Pair the 25 pre-dedup diagnostics with their exact-dedup mates."""
    keys = ["part_slug", "base_config_id"]
    metrics = ["val_pearson", "val_spearman", "val_cod_r2", "val_mse", "val_rmse", "best_epoch"]
    exact = results.loc[results["run_kind"].eq("exact_replay"), keys + metrics + ["run_id"]]
    pre = results.loc[
        results["run_kind"].eq("pre_dedup_calibration"), keys + metrics + ["run_id"]
    ]
    paired = pre.merge(exact, on=keys, suffixes=("_pre_dedup", "_dedup"), validate="one_to_one")
    for metric in metrics:
        paired[f"delta_dedup_minus_pre_{metric}"] = (
            paired[f"{metric}_dedup"] - paired[f"{metric}_pre_dedup"]
        )
    return paired.sort_values(keys).reset_index(drop=True)


def _config_matrix(part_results: pd.DataFrame) -> pd.DataFrame:
    hp_columns = [column for column in part_results.columns if column.startswith("hp_")]
    raw = part_results[hp_columns].copy()
    numeric = []
    categorical = []
    for column in raw:
        nonnull = raw[column].dropna()
        if nonnull.nunique() <= 1:
            continue
        converted = pd.to_numeric(nonnull, errors="coerce")
        if converted.notna().all():
            numeric.append(column)
        else:
            categorical.append(column)
    pieces = []
    if numeric:
        values = raw[numeric].apply(pd.to_numeric, errors="coerce")
        for column in values:
            if column in {"hp_lr", "hp_weight_decay", "hp_eps"}:
                positive = values[column] > 0
                values.loc[positive, column] = np.log10(values.loc[positive, column])
        values = values.fillna(values.median())
        scale = values.std(ddof=0).replace(0, 1)
        pieces.append((values - values.mean()) / scale)
    if categorical:
        pieces.append(pd.get_dummies(raw[categorical].fillna("<NA>").astype(str), dtype=float))
    if not pieces:
        return pd.DataFrame(index=part_results.index)
    matrix = pd.concat(pieces, axis=1)
    return matrix.loc[:, matrix.nunique() > 1]


def _maximin_pick(
    candidates: pd.DataFrame,
    matrix: pd.DataFrame,
    selected_indices: list[int],
) -> int:
    remaining = [idx for idx in candidates.index if idx not in selected_indices]
    if not remaining:
        raise ValueError("No candidates remain for diversity selection.")
    if matrix.empty or not selected_indices:
        return int(candidates.loc[remaining, "val_pearson"].idxmax())
    selected = matrix.loc[selected_indices].to_numpy(float)
    scored = []
    for idx in remaining:
        vector = matrix.loc[idx].to_numpy(float)
        min_distance = float(np.sqrt(np.mean((selected - vector) ** 2, axis=1)).min())
        scored.append((min_distance, float(candidates.loc[idx, "val_pearson"]), int(idx)))
    return max(scored)[2]


def select_stage2_candidates(
    results: pd.DataFrame,
    intron_subset_run_metrics: pd.DataFrame | None = None,
    intron_decomposition: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Produce a deterministic, reviewable draft of the plan's 10-per-part rule.

    This is a nomination table, not authorization to launch Stage 2. Prediction
    bootstrap intervals and scientific judgment should be reviewed before the
    table is frozen.
    """
    exact = results.loc[results["run_kind"].eq("exact_replay")].copy()
    selections = []
    for part, frame in exact.groupby("part_slug", sort=True):
        frame = frame.sort_values(
            ["val_pearson", "val_cod_r2", "val_mse", "base_config_id"],
            ascending=[False, False, True, True],
        ).copy()
        frame["stage1_pearson_rank"] = np.arange(1, len(frame) + 1)
        quartile_cut = frame["val_pearson"].quantile(0.75)
        top_quartile = frame.loc[frame["val_pearson"].ge(quartile_cut)]
        matrix = _config_matrix(frame)

        chosen: list[int] = []
        reasons: dict[int, str] = {}
        for idx in frame.head(6).index:
            chosen.append(int(idx))
            reasons[int(idx)] = "top6_val_pearson"

        for number in range(1, 3):
            idx = _maximin_pick(top_quartile, matrix, chosen)
            chosen.append(idx)
            reasons[idx] = f"top_quartile_hyperparameter_diversity_{number}"

        if (
            part == "intron"
            and intron_subset_run_metrics is not None
            and intron_decomposition is not None
        ):
            decomp_map = intron_decomposition.set_index("run_id")[
                "within_subset_centered_pearson"
            ]
            worst_map = intron_subset_run_metrics.groupby("run_id")[
                "prediction_pearson"
            ].min()
            frame["inferred_within_centered_pearson"] = frame["run_id"].map(decomp_map)
            frame["inferred_worst_stratum_pearson"] = frame["run_id"].map(worst_map)
            top_quartile = frame.loc[frame["val_pearson"].ge(quartile_cut)]

            remaining = top_quartile.loc[~top_quartile.index.isin(chosen)]
            within_idx = int(
                remaining.sort_values(
                    ["inferred_within_centered_pearson", "val_pearson"], ascending=False
                ).index[0]
            )
            chosen.append(within_idx)
            reasons[within_idx] = "intron_inferred_within_stratum_complement"

            remaining = top_quartile.loc[~top_quartile.index.isin(chosen)]
            worst_idx = int(
                remaining.sort_values(
                    ["inferred_worst_stratum_pearson", "val_pearson"], ascending=False
                ).index[0]
            )
            chosen.append(worst_idx)
            reasons[worst_idx] = "intron_inferred_worst_stratum_complement"
        else:
            remaining = top_quartile.loc[~top_quartile.index.isin(chosen)]
            if remaining.empty:
                remaining = frame.loc[~frame.index.isin(chosen)]
            rmse_idx = int(
                remaining.sort_values(["val_rmse", "val_pearson"], ascending=[True, False]).index[0]
            )
            chosen.append(rmse_idx)
            reasons[rmse_idx] = "strong_validation_best_complementary_rmse"

            remaining = top_quartile.loc[~top_quartile.index.isin(chosen)]
            if remaining.empty:
                remaining = frame.loc[~frame.index.isin(chosen)]
            cod_idx = int(
                remaining.sort_values(["val_cod_r2", "val_pearson"], ascending=False).index[0]
            )
            chosen.append(cod_idx)
            reasons[cod_idx] = "strong_validation_best_complementary_cod_r2"

        if part == "utr5" and frame["architecture"].nunique() > 1:
            architecture_leaders = frame.groupby("architecture", sort=False).head(1)
            leader = architecture_leaders.iloc[0]
            alternate = architecture_leaders.loc[
                architecture_leaders["architecture"].ne(leader["architecture"])
            ].sort_values("val_pearson", ascending=False).iloc[0]
            near_tied = float(leader["val_pearson"] - alternate["val_pearson"]) <= 0.01
            alternate_present = alternate.name in chosen
            if near_tied and not alternate_present:
                replaced = chosen[-1]
                chosen[-1] = int(alternate.name)
                reasons.pop(replaced, None)
                reasons[int(alternate.name)] = "5utr_near_tied_architecture_guard"

        if len(chosen) != 10 or len(set(chosen)) != 10:
            raise AssertionError(f"Selection for {part} is not 10 unique rows: {chosen}")
        selected = frame.loc[chosen].copy()
        selected["selection_reason"] = [reasons[int(idx)] for idx in selected.index]
        selected["selection_order"] = np.arange(1, 11)
        selections.append(selected)

    selected = pd.concat(selections).sort_values(["part_slug", "selection_order"])
    columns = [
        "part_slug",
        "selection_order",
        "selection_reason",
        "stage1_pearson_rank",
        "architecture",
        "base_config_id",
        "run_id",
        "run_url",
        "val_pearson",
        "val_spearman",
        "val_cod_r2",
        "val_mse",
        "val_rmse",
        "best_epoch",
        "train_pearson",
        "train_minus_val_pearson",
        "source_kind",
        "prediction_path",
        "dataset_sha256",
        "split_manifest_sha256",
        "selected_row_hash",
        "normalization_row_id_hash",
        "row_fingerprint",
    ]
    return selected[columns].reset_index(drop=True)


def bootstrap_pearson_intervals(
    results: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 1701,
) -> pd.DataFrame:
    """Row-bootstrap Pearson intervals for a supplied subset of runs."""
    rng = np.random.default_rng(seed)
    records = []
    for _, row in results.iterrows():
        predictions = load_predictions(row)
        target = predictions["target_processed"].to_numpy(float)
        pred = predictions["prediction_processed"].to_numpy(float)
        draws = np.empty(n_boot, dtype=float)
        for draw in range(n_boot):
            index = rng.integers(0, len(target), len(target))
            draws[draw] = np.corrcoef(target[index], pred[index])[0, 1]
        records.append(
            {
                "part_slug": row["part_slug"],
                "architecture": row["architecture"],
                "base_config_id": row["base_config_id"],
                "run_id": row["run_id"],
                "val_pearson": row["val_pearson"],
                "pearson_boot_mean": float(np.nanmean(draws)),
                "pearson_ci_low": float(np.nanquantile(draws, 0.025)),
                "pearson_ci_high": float(np.nanquantile(draws, 0.975)),
            }
        )
    return pd.DataFrame(records)


def hyperparameter_associations(results: pd.DataFrame) -> pd.DataFrame:
    """Exploratory univariate Spearman associations, never a causal analysis."""
    exact = results.loc[results["run_kind"].eq("exact_replay")]
    records = []
    for (part, architecture), frame in exact.groupby(["part_slug", "architecture"]):
        for column in [name for name in frame.columns if name.startswith("hp_")]:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna() & frame["val_pearson"].notna()
            if valid.sum() < 20 or values.loc[valid].nunique() < 3:
                continue
            rho, pvalue = spearmanr(values.loc[valid], frame.loc[valid, "val_pearson"])
            records.append(
                {
                    "part_slug": part,
                    "architecture": architecture,
                    "hyperparameter": column[3:] if column.startswith("hp_") else column,
                    "n": int(valid.sum()),
                    "spearman_rho": float(rho),
                    "pvalue_unadjusted": float(pvalue),
                    "abs_spearman_rho": abs(float(rho)),
                }
            )
    return pd.DataFrame(records).sort_values(
        ["part_slug", "architecture", "abs_spearman_rho"], ascending=[True, True, False]
    )


def hpo_saturation_analysis(
    results: pd.DataFrame,
    sample_sizes: Iterable[int] = (10, 20, 40, 80, 120),
    draws: int = 1000,
    seed: int = 1701,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate best-of-N saturation and summarize the near-best plateau.

    Random config subsampling is descriptive because the historical candidates
    were not drawn from one uniform factorial distribution. It is nevertheless
    useful for distinguishing a broad plateau from a lone fold-0 outlier.
    """
    exact = results.loc[results["run_kind"].eq("exact_replay")]
    rng = np.random.default_rng(seed)
    curves = []
    plateaus = []
    for part, frame in exact.groupby("part_slug", sort=True):
        values = frame["val_pearson"].dropna().to_numpy(float)
        ordered = np.sort(values)[::-1]
        best = float(ordered[0])
        plateaus.append(
            {
                "part_slug": part,
                "configs": len(values),
                "best_val_pearson": best,
                "within_0.005_of_best": int((values >= best - 0.005).sum()),
                "within_0.01_of_best": int((values >= best - 0.01).sum()),
                "within_0.02_of_best": int((values >= best - 0.02).sum()),
                "within_0.05_of_best": int((values >= best - 0.05).sum()),
                "top1_minus_top10": float(ordered[0] - ordered[min(9, len(ordered) - 1)]),
                "top1_minus_top25": float(ordered[0] - ordered[min(24, len(ordered) - 1)]),
            }
        )
        for count in sample_sizes:
            if count > len(values):
                continue
            maxima = np.asarray(
                [rng.choice(values, size=count, replace=False).max() for _ in range(draws)]
            )
            curves.append(
                {
                    "part_slug": part,
                    "sampled_configs": count,
                    "expected_best_pearson": float(maxima.mean()),
                    "best_ci_low": float(np.quantile(maxima, 0.025)),
                    "best_ci_high": float(np.quantile(maxima, 0.975)),
                    "observed_full_best": best,
                }
            )
    return pd.DataFrame(plateaus), pd.DataFrame(curves)


def assign_inferred_intron_subsets(frame: pd.DataFrame, sequence_column: str = "Intron") -> pd.DataFrame:
    """Assign mutually exclusive mask strata using most-specific-first precedence.

    The collaborator-provided masks are nested: mask 1 is contained in mask 2,
    and the all-N mask contains every 80-nt sequence. Therefore these labels are
    inferred sequence strata, not recoverable experimental provenance. True
    synthesis-pool labels should replace them if they become available.
    """
    compiled = {
        name: re.compile("^" + "".join(IUPAC_REGEX[base] for base in mask) + "$")
        for name, mask in INTRON_MASKS.items()
    }
    updated = frame.copy()
    mask1 = updated[sequence_column].astype(str).map(
        lambda sequence: bool(compiled["mask1_specific"].match(sequence))
    )
    mask2 = updated[sequence_column].astype(str).map(
        lambda sequence: bool(compiled["mask2"].match(sequence))
    )
    mask3 = updated[sequence_column].astype(str).map(
        lambda sequence: bool(compiled["mask3"].match(sequence))
    )
    if not mask3.all():
        invalid = int((~mask3).sum())
        raise ValueError(
            f"Found {invalid} Intron sequences that are not exact-80 canonical DNA; "
            "they cannot be assigned to the all-N residual stratum."
        )
    if (mask1 & ~mask2).any():
        raise AssertionError("The supplied intron mask 1 is not nested inside mask 2.")
    updated["inferred_intron_subset"] = np.where(
        mask1,
        "mask1_specific",
        np.where(mask2, "mask2_not_mask1", "mask3_residual"),
    )
    return updated


def analyze_intron_subsets(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit inferred Intron mask strata and every Stage 1 Intron prediction table."""
    intron_runs = results.loc[
        results["run_kind"].eq("exact_replay") & results["part_slug"].eq("intron")
    ].copy()
    if intron_runs.empty:
        raise ValueError("No exact-replay Intron rows were found.")
    dataset_paths = intron_runs["dataset_path"].dropna().unique()
    if len(dataset_paths) != 1:
        raise ValueError(f"Expected one Intron dataset path; found {len(dataset_paths)}")
    dataset = pd.read_csv(dataset_paths[0], sep="\t")
    dataset = assign_inferred_intron_subsets(dataset)
    split_manifest_paths = intron_runs["split_manifest_path"].dropna().unique()
    if len(split_manifest_paths) != 1:
        raise ValueError(
            f"Expected one Intron split manifest path; found {len(split_manifest_paths)}"
        )
    split_manifest = json.loads(Path(split_manifest_paths[0]).read_text(encoding="utf-8"))
    assignments = pd.DataFrame(split_manifest["assignments"])[
        ["construct_id", "partition", "development_fold"]
    ]
    dataset = dataset.merge(assignments, on="construct_id", how="left", validate="one_to_one")
    if dataset["partition"].isna().any():
        raise ValueError("The Intron dataset contains IDs absent from the split manifest.")

    # Target summaries used during model selection must not aggregate the frozen
    # audit partition. Sequence-only compatibility can be counted over all rows,
    # but the visible target/barcode summary below is explicitly non-audit.
    descriptive_dataset = dataset.loc[dataset["partition"].ne("audit_test")].copy()
    subset_summary = (
        descriptive_dataset.groupby("inferred_intron_subset")
        .agg(
            constructs=("construct_id", "size"),
            median_barcodes=("n_barcodes", "median"),
            mean_barcodes=("n_barcodes", "mean"),
            hq_constructs=("n_barcodes", lambda values: int((values >= 8).sum())),
            mean_log2_target=("log2_RNA_DNA", "mean"),
            sd_log2_target=("log2_RNA_DNA", "std"),
        )
        .reset_index()
    )
    subset_summary.insert(0, "analysis_scope", "non_audit_dataset")

    records = []
    decomposition = []
    subset_map = dataset[["construct_id", "inferred_intron_subset"]]
    for _, run in intron_runs.iterrows():
        prediction = load_predictions(run).merge(
            subset_map, on="construct_id", how="left", validate="one_to_one"
        )
        if prediction["inferred_intron_subset"].isna().any():
            raise ValueError(f"Unmatched Intron construct IDs for run {run['run_id']}")
        for subset, group in prediction.groupby("inferred_intron_subset"):
            metrics = prediction_metrics(group)
            records.append(
                {
                    "run_id": run["run_id"],
                    "base_config_id": run["base_config_id"],
                    "best_epoch": run["best_epoch"],
                    "overall_val_pearson": run["val_pearson"],
                    "inferred_intron_subset": subset,
                    "validation_rows": len(group),
                    **metrics,
                }
            )
        target = prediction["target_processed"].to_numpy(float)
        predicted = prediction["prediction_processed"].to_numpy(float)
        target_group_mean = prediction.groupby("inferred_intron_subset")[
            "target_processed"
        ].transform("mean").to_numpy(float)
        prediction_group_mean = prediction.groupby("inferred_intron_subset")[
            "prediction_processed"
        ].transform("mean").to_numpy(float)
        within_target = target - target_group_mean
        within_prediction = predicted - prediction_group_mean
        development_fold = int(run["development_fold"])
        fold_training = dataset.loc[
            dataset["partition"].eq("train_only")
            | (
                dataset["partition"].eq("development")
                & dataset["development_fold"].ne(development_fold)
            )
        ]
        train_stratum_means = fold_training.groupby("inferred_intron_subset")[
            "log2_RNA_DNA"
        ].mean()
        train_fitted_baseline = prediction["inferred_intron_subset"].map(
            train_stratum_means
        ).to_numpy(float)
        raw_target = prediction["log2_RNA_DNA"].to_numpy(float)
        raw_prediction = prediction["prediction_raw"].to_numpy(float)
        oracle_validation_baseline = prediction.groupby("inferred_intron_subset")[
            "log2_RNA_DNA"
        ].transform("mean").to_numpy(float)
        decomposition.append(
            {
                "run_id": run["run_id"],
                "base_config_id": run["base_config_id"],
                "best_epoch": run["best_epoch"],
                "overall_val_pearson": run["val_pearson"],
                "within_stratum_centered_pearson": float(pearsonr(within_target, within_prediction)[0]),
                # Backward-compatible alias retained for the Stage 1 selection notebook.
                "within_subset_centered_pearson": float(pearsonr(within_target, within_prediction)[0]),
                "oracle_validation_stratum_mean_baseline_pearson": float(
                    pearsonr(raw_target, oracle_validation_baseline)[0]
                ),
                "train_fitted_stratum_mean_baseline_pearson": float(
                    pearsonr(raw_target, train_fitted_baseline)[0]
                ),
                "oracle_validation_stratum_mean_baseline_rmse_raw": float(
                    np.sqrt(np.mean((raw_target - oracle_validation_baseline) ** 2))
                ),
                "train_fitted_stratum_mean_baseline_rmse_raw": float(
                    np.sqrt(np.mean((raw_target - train_fitted_baseline) ** 2))
                ),
                "model_rmse_raw": float(
                    np.sqrt(np.mean((raw_target - raw_prediction) ** 2))
                ),
                # Deprecated aliases: these use validation-derived (oracle) means.
                "subset_mean_only_baseline_pearson": float(pearsonr(target, target_group_mean)[0]),
                "subset_mean_only_baseline_mse": float(np.mean((target - target_group_mean) ** 2)),
                "model_mse": float(np.mean((target - predicted) ** 2)),
                "between_subset_target_variance_fraction": float(
                    np.var(target_group_mean) / np.var(target)
                ),
            }
        )
    return subset_summary, pd.DataFrame(records), pd.DataFrame(decomposition)


def find_local_wandb_file(run_id: str, wandb_root: str | Path | None = None) -> Path:
    root = Path(wandb_root) if wandb_root else LEARN_ROOT / "wandb"
    matches = list(root.glob(f"run-*-{run_id}/run-{run_id}.wandb"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one local W&B file for {run_id}; found {len(matches)}")
    return matches[0]


def write_curated_tables(
    output_dir: str | Path,
    results: pd.DataFrame,
    selection: pd.DataFrame,
    calibration: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> list[Path]:
    """Write compact, reviewable evidence tables used to freeze Stage 2 inputs."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "stage1_exact_replay_metrics.csv": results.loc[
            results["run_kind"].eq("exact_replay"),
            [
                "part_slug",
                "architecture",
                "base_config_id",
                "run_id",
                "run_url",
                "source_kind",
                "val_pearson",
                "val_spearman",
                "val_cod_r2",
                "val_mse",
                "val_rmse",
                "best_epoch",
                "train_pearson",
                "train_mse",
                "prediction_path",
                "row_fingerprint",
            ],
        ],
        "stage1_pre_dedup_calibration_pairs.csv": calibration,
        "stage2_candidate_selection_draft.csv": selection,
        "stage2_candidate_bootstrap_intervals.csv": bootstrap,
    }
    written = []
    for filename, frame in tables.items():
        path = root / filename
        frame.to_csv(path, index=False)
        written.append(path)
    return written
