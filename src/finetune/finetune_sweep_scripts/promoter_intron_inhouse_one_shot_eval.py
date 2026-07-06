#!/usr/bin/env python3
"""
One-shot in-house evaluation for pretrained promoter and intron part models.

This runner is intentionally diagnostic. It scores the best available public
pretrained promoter/intron checkpoints on the MattLee Lib1 in-house part tables
and writes barcode-stratified metrics plus prediction files under an ignored
`src/finetune/learning_curve/` result directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boda.model  # noqa: E402
from boda.common import utils  # noqa: E402


DEFAULT_OUTDIR = (
    REPO_ROOT
    / "src"
    / "finetune"
    / "learning_curve"
    / "promoter_intron_pretrained_inhouse_eval_may2026"
)
DEFAULT_RUNS_CSV = REPO_ROOT / "src" / "learn" / "run_registry" / "runs.csv"
DEFAULT_PROMOTER_PATH = (
    Path("/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1")
    / "single_part_variant_level"
    / "promoters"
    / "L1_final_fastqs1-5_sublibrary_Promoter_subset.csv"
)
DEFAULT_INTRON_PATH = (
    Path("/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1")
    / "single_part_variant_level"
    / "introns"
    / "L1_final_fastqs1-5_sublibrary_Intron_subset.csv"
)


@dataclass(frozen=True)
class PartEvalSpec:
    part_type: str
    task_family: str
    target_family: str
    data_path: Path
    sequence_column: str
    transform_policy: str
    notes: str


DEFAULT_SPECS = [
    PartEvalSpec(
        part_type="promoter",
        task_family="promoter",
        target_family="deboer_core",
        data_path=DEFAULT_PROMOTER_PATH,
        sequence_column="Promoter",
        transform_policy="right_pad_N",
        notes=(
            "PromoterDataModule pads shorter promoter sequences on the right "
            "with neutral N bases to the model input length."
        ),
    ),
    PartEvalSpec(
        part_type="intron",
        task_family="introns",
        target_family="seelig_2015_a5ss_sd1",
        data_path=DEFAULT_INTRON_PATH,
        sequence_column="Intron",
        transform_policy="center_pad_N",
        notes=(
            "Diagnostic transfer only: Seelig A5SS predicts splice-site usage "
            "from 101 nt sequences, while the in-house target is RNA/DNA and "
            "the in-house introns are mostly 80 nt."
        ),
    ),
]


def safe_extract(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([str(target_root), str(member_path)]) != str(target_root):
            raise RuntimeError(f"Unsafe path in tarball: {member.name}")
    tar.extractall(str(target_dir))


def load_checkpoint_from_tar(artifact_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(str(artifact_path)) as tar:
            safe_extract(tar, tmp_path)
        return torch.load(tmp_path / "artifacts" / "torch_checkpoint.pt", map_location=map_location)


def namespace_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    if isinstance(value, dict):
        return value.copy()
    raise TypeError(f"Cannot convert {type(value)} to dict")


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> torch.nn.Module:
    model_cls = getattr(boda.model, str(checkpoint["model_module"]))
    model_hparams = namespace_to_dict(checkpoint["model_hparams"])
    model = model_cls(**model_hparams)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def clean_sequence(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().upper()


def is_valid_dna_or_n(seq: str) -> bool:
    return bool(seq) and set(seq).issubset({"A", "C", "G", "T", "N"})


def gc_fraction(seq: str) -> float:
    acgt = [base for base in seq if base in {"A", "C", "G", "T"}]
    if not acgt:
        return float("nan")
    return float(sum(base in {"G", "C"} for base in acgt) / len(acgt))


def transform_sequence(seq: str, target_len: int, policy: str) -> str:
    seq = clean_sequence(seq)
    if len(seq) == target_len:
        return seq
    if len(seq) > target_len:
        if policy == "center_pad_N":
            start = (len(seq) - target_len) // 2
            return seq[start : start + target_len]
        return seq[:target_len]

    missing = target_len - len(seq)
    if policy == "right_pad_N":
        return seq + ("N" * missing)
    if policy == "left_pad_N":
        return ("N" * missing) + seq
    if policy == "center_pad_N":
        left = missing // 2
        right = missing - left
        return ("N" * left) + seq + ("N" * right)
    raise ValueError(f"Unknown transform policy: {policy}")


def add_barcode_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    bc = pd.to_numeric(out.get("number_of_barcodes"), errors="coerce")
    out["barcode_count"] = bc
    out["barcode_bin"] = pd.cut(
        bc,
        bins=[-np.inf, 1, 3, 7, 15, np.inf],
        labels=["bc_1", "bc_2_3", "bc_4_7", "bc_8_15", "bc_16_plus"],
    ).astype("string")
    out.loc[bc.isna(), "barcode_bin"] = "bc_missing"
    return out


def select_best_run(
    runs_csv: Path,
    task_family: str,
    target_family: str,
    metric_column: str = "best_metric_value",
) -> dict[str, Any]:
    runs = pd.read_csv(runs_csv)
    sub = runs[
        (runs["task_family"].astype(str) == task_family)
        & (runs["target_family"].astype(str) == target_family)
    ].copy()
    if sub.empty:
        raise ValueError(f"No runs found for {task_family}/{target_family} in {runs_csv}")

    sub[metric_column] = pd.to_numeric(sub[metric_column], errors="coerce")
    sub["artifact_exists"] = sub["artifact_path"].map(lambda p: Path(str(p)).is_file())
    usable = sub[sub["artifact_exists"] & sub[metric_column].notna()].copy()
    if usable.empty:
        raise ValueError(f"No usable artifact rows found for {task_family}/{target_family}")

    row = usable.sort_values(metric_column, ascending=False).iloc[0]
    return row.to_dict()


def prepare_inhouse_table(spec: PartEvalSpec, input_len: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(spec.data_path)
    if spec.sequence_column not in raw.columns:
        raise ValueError(f"{spec.data_path} is missing sequence column {spec.sequence_column!r}")

    df = raw.copy()
    df["row_id"] = np.arange(len(df), dtype=int)
    df["part_type"] = spec.part_type
    df["source_path"] = str(spec.data_path)
    df["sequence_original"] = df[spec.sequence_column].map(clean_sequence)
    df["sequence_len"] = df["sequence_original"].str.len()
    df["valid_original_dna"] = df["sequence_original"].map(is_valid_dna_or_n)
    df["gc_fraction"] = df["sequence_original"].map(gc_fraction)
    df["transform_policy"] = spec.transform_policy
    df["model_input_len"] = int(input_len)
    df["sequence_transformed"] = df["sequence_original"].map(
        lambda seq: transform_sequence(seq, input_len, spec.transform_policy)
    )
    df["transformed_len"] = df["sequence_transformed"].str.len()
    df["valid_transformed_dna"] = df["sequence_transformed"].map(is_valid_dna_or_n)
    df["transform_action"] = np.select(
        [
            df["sequence_len"] == input_len,
            df["sequence_len"] < input_len,
            df["sequence_len"] > input_len,
        ],
        ["exact", "pad", "truncate"],
        default="unknown",
    )

    df["RNA/DNA"] = pd.to_numeric(df.get("RNA/DNA"), errors="coerce")
    df["log2_RNA_DNA"] = np.log2(df["RNA/DNA"].where(df["RNA/DNA"] > 0))
    for col in ["number_of_barcodes", "DNA_bc_counts_sum", "RNA_bc_counts_sum"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_barcode_bin(df)

    usable = df[
        df["valid_original_dna"]
        & df["valid_transformed_dna"]
        & (df["transformed_len"] == input_len)
    ].copy()

    audit = {
        "part_type": spec.part_type,
        "source_path": str(spec.data_path),
        "sequence_column": spec.sequence_column,
        "target_columns": ["RNA/DNA", "log2_RNA_DNA"],
        "transform_policy": spec.transform_policy,
        "model_input_len": int(input_len),
        "notes": spec.notes,
        "n_raw_rows": int(len(df)),
        "n_usable_rows": int(len(usable)),
        "n_invalid_original": int((~df["valid_original_dna"]).sum()),
        "n_invalid_transformed": int((~df["valid_transformed_dna"]).sum()),
        "sequence_len_counts": {
            str(k): int(v) for k, v in df["sequence_len"].value_counts().sort_index().items()
        },
        "transform_action_counts": {
            str(k): int(v) for k, v in df["transform_action"].value_counts().sort_index().items()
        },
        "barcode_count_summary": {
            "min": float(df["barcode_count"].min(skipna=True)),
            "median": float(df["barcode_count"].median(skipna=True)),
            "max": float(df["barcode_count"].max(skipna=True)),
        },
    }
    return usable.reset_index(drop=True), audit


def predict_sequences(
    model: torch.nn.Module,
    sequences: Iterable[str],
    device: str,
    batch_size: int,
) -> np.ndarray:
    tensors = torch.stack([utils.dna2tensor(seq) for seq in sequences], dim=0)
    loader = DataLoader(TensorDataset(tensors), batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for (batch,) in loader:
            pred = model(batch.to(device))
            predictions.append(pred.detach().cpu().numpy())
    out = np.concatenate(predictions, axis=0)
    if out.ndim == 1:
        out = out[:, None]
    return out


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    if np.nanstd(y_true) < 1e-12 or np.nanstd(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    n = int(len(y))
    if n == 0:
        return {"n": 0, "pearson": np.nan, "spearman": np.nan, "cod_r2": np.nan, "rmse": np.nan, "mae": np.nan}

    pearson = safe_corr(y, p)
    spearman = safe_corr(pd.Series(y).rank(method="average").to_numpy(), pd.Series(p).rank(method="average").to_numpy())
    residual = y - p
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    cod_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n": n,
        "pearson": pearson,
        "spearman": spearman,
        "cod_r2": cod_r2,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def metric_subsets(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    bc = pd.to_numeric(frame["barcode_count"], errors="coerce")
    subsets: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=frame.index))]
    for label in ["bc_1", "bc_2_3", "bc_4_7", "bc_8_15", "bc_16_plus"]:
        subsets.append((label, frame["barcode_bin"].eq(label)))
    for threshold in [2, 4, 8, 16]:
        subsets.append((f"bc_ge_{threshold}", bc >= threshold))
    return subsets


def evaluate_predictions(pred_df: pd.DataFrame, run_row: dict[str, Any]) -> pd.DataFrame:
    rows = []
    prediction_columns = [c for c in pred_df.columns if c.startswith("pred_output_")]
    for target_column in ["log2_RNA_DNA", "RNA/DNA"]:
        for prediction_column in prediction_columns:
            for subset_id, subset_mask in metric_subsets(pred_df):
                subset = pred_df[subset_mask].copy()
                metrics = regression_metrics(subset[target_column], subset[prediction_column])
                rows.append(
                    {
                        "part_type": pred_df["part_type"].iloc[0],
                        "run_id": run_row.get("run_id", ""),
                        "target_family": run_row.get("target_family", ""),
                        "model_module": run_row.get("model_module", ""),
                        "model_public_metric": run_row.get("best_metric_value", np.nan),
                        "target_column": target_column,
                        "prediction_column": prediction_column,
                        "subset_id": subset_id,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_part(
    spec: PartEvalSpec,
    runs_csv: Path,
    outdir: Path,
    device: str,
    batch_size: int,
) -> dict[str, Path]:
    run_row = select_best_run(runs_csv, spec.task_family, spec.target_family)
    artifact_path = Path(str(run_row["artifact_path"]))
    checkpoint = load_checkpoint_from_tar(artifact_path, map_location="cpu")
    input_len = int(namespace_to_dict(checkpoint["model_hparams"])["input_len"])
    model = build_model_from_checkpoint(checkpoint, device=device)

    eval_df, audit = prepare_inhouse_table(spec, input_len=input_len)
    pred = predict_sequences(model, eval_df["sequence_transformed"].tolist(), device=device, batch_size=batch_size)

    pred_df = eval_df[
        [
            "row_id",
            "part_type",
            "sequence_original",
            "sequence_transformed",
            "sequence_len",
            "transformed_len",
            "gc_fraction",
            "transform_policy",
            "transform_action",
            "RNA/DNA",
            "log2_RNA_DNA",
            "number_of_barcodes",
            "barcode_count",
            "barcode_bin",
            "DNA_bc_counts_sum",
            "RNA_bc_counts_sum",
        ]
    ].copy()
    for idx in range(pred.shape[1]):
        pred_df[f"pred_output_{idx}"] = pred[:, idx]

    pred_df["run_id"] = run_row.get("run_id", "")
    pred_df["candidate_id"] = f"{spec.part_type}__{run_row.get('run_id', '')}"
    pred_df["artifact_path"] = str(artifact_path)
    pred_df["source_public_metric_name"] = run_row.get("best_metric_name", "")
    pred_df["source_public_metric_value"] = run_row.get("best_metric_value", np.nan)

    metrics = evaluate_predictions(pred_df, run_row)

    part_dir = outdir / spec.part_type / str(run_row.get("run_id", "unknown"))
    part_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = part_dir / "predictions.csv"
    metrics_path = part_dir / "one_shot_metrics.csv"
    audit_path = part_dir / "data_audit.json"
    model_card_path = part_dir / "model_card.md"

    pred_df.to_csv(prediction_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    audit.update(
        {
            "run_row": {
                key: run_row.get(key, "")
                for key in [
                    "timestamp",
                    "run_id",
                    "task_family",
                    "target_family",
                    "comparison_group",
                    "data_module",
                    "model_module",
                    "graph_module",
                    "best_metric_name",
                    "best_metric_value",
                    "artifact_path",
                    "config_path",
                ]
            },
            "prediction_path": str(prediction_path),
            "metrics_path": str(metrics_path),
        }
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    model_card_path.write_text(render_model_card(spec, run_row, audit, metrics), encoding="utf-8")

    return {
        "part_dir": part_dir,
        "prediction_path": prediction_path,
        "metrics_path": metrics_path,
        "audit_path": audit_path,
        "model_card_path": model_card_path,
    }


def render_model_card(
    spec: PartEvalSpec,
    run_row: dict[str, Any],
    audit: dict[str, Any],
    metrics: pd.DataFrame,
) -> str:
    primary = metrics[
        (metrics["target_column"] == "log2_RNA_DNA")
        & (metrics["prediction_column"] == "pred_output_0")
        & (metrics["subset_id"].isin(["all", "bc_ge_4", "bc_ge_8", "bc_ge_16"]))
    ].copy()
    lines = [
        f"# {spec.part_type.title()} One-Shot In-House Evaluation",
        "",
        "## Source Model",
        f"- run_id: `{run_row.get('run_id', '')}`",
        f"- task_family: `{run_row.get('task_family', '')}`",
        f"- target_family: `{run_row.get('target_family', '')}`",
        f"- model_module: `{run_row.get('model_module', '')}`",
        f"- public_metric: `{run_row.get('best_metric_name', '')}={run_row.get('best_metric_value', '')}`",
        f"- artifact_path: `{run_row.get('artifact_path', '')}`",
        "",
        "## In-House Data",
        f"- source_path: `{spec.data_path}`",
        f"- sequence_column: `{spec.sequence_column}`",
        f"- transform_policy: `{spec.transform_policy}`",
        f"- model_input_len: `{audit['model_input_len']}`",
        f"- usable_rows: `{audit['n_usable_rows']}` of `{audit['n_raw_rows']}`",
        "",
        "## Interpretation",
        spec.notes,
        "",
        "## Primary Diagnostic Metrics",
        primary.to_markdown(index=False) if len(primary) else "No primary metrics available.",
        "",
    ]
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_default_evaluations(
    outdir: Path = DEFAULT_OUTDIR,
    runs_csv: Path = DEFAULT_RUNS_CSV,
    device: str | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    paths_by_part = {}
    for spec in DEFAULT_SPECS:
        paths = evaluate_part(spec, runs_csv=runs_csv, outdir=outdir, device=device, batch_size=batch_size)
        paths_by_part[spec.part_type] = paths
        metrics = pd.read_csv(paths["metrics_path"])
        audit = json.loads(paths["audit_path"].read_text(encoding="utf-8"))
        records.append(
            {
                "spec": json_safe(asdict(spec)),
                "paths": {k: str(v) for k, v in paths.items()},
                "audit": json_safe(audit),
            }
        )

    combined_metrics = []
    for paths in paths_by_part.values():
        combined_metrics.append(pd.read_csv(paths["metrics_path"]))
    combined_metrics_df = pd.concat(combined_metrics, ignore_index=True)
    combined_metrics_path = outdir / "combined_one_shot_metrics.csv"
    combined_metrics_df.to_csv(combined_metrics_path, index=False)

    manifest = {
        "device": device,
        "batch_size": int(batch_size),
        "runs_csv": str(runs_csv),
        "outdir": str(outdir),
        "combined_metrics_path": str(combined_metrics_path),
        "parts": records,
    }
    manifest_path = outdir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    return {"manifest_path": manifest_path, "combined_metrics_path": combined_metrics_path, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--runs_csv", type=Path, default=DEFAULT_RUNS_CSV)
    parser.add_argument("--device", default=None, help="cpu, cuda, or blank for auto")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_default_evaluations(
        outdir=args.outdir,
        runs_csv=args.runs_csv,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps({k: str(v) for k, v in result.items() if k != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
