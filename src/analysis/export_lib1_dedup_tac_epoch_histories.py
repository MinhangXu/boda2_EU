#!/usr/bin/env python3
"""Export presentation-ready epoch histories for the frozen Lib1 policies.

The export is intentionally development-only: it includes train and validation
metrics but never test/final-test trajectories.  Selected weighted-policy
histories are read from the local W&B binary files; the selected Enhancer
unweighted policy and the controlled Enhancer unfreeze-scope comparison are
read from the previously frozen Stage-2 local-history export.

Run in the environment that can read the local W&B protobuf files::

    conda run -n boda_env python \
      src/analysis/export_lib1_dedup_tac_epoch_histories.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

SELECTED_POLICIES_PATH = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026"
    / "stage3_selected_part_policies.csv"
)
STAGE2_HISTORIES_PATH = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting"
    / "stage2_learning_histories.tsv.gz"
)
STAGE2_OOF_PATH = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_stage2_july2026"
    / "stage2_oof_metrics.csv"
)
STAGE2_MANIFEST_PATH = (
    REPO_ROOT
    / "src/learn/outputs/hpo_manifests"
    / "lib1_dedup_stage2_july2026__run_manifest.jsonl"
)
RUN_REGISTRY_PATH = REPO_ROOT / "src/learn/run_registry/runs.csv"
WANDB_ROOT = REPO_ROOT / "src/learn/wandb"
WANDB_EXPORTER_PATH = REPO_ROOT / "src/learn/export_wandb_history.py"

OUTPUT_ROOT = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/tables"
)
SELECTED_OUTPUT_PATH = OUTPUT_ROOT / "selected_policy_epoch_histories.tsv"
ENHANCER_SCOPE_OUTPUT_PATH = (
    OUTPUT_ROOT / "enhancer_k562_rc_on_unfreeze_scope_epoch_histories.tsv"
)
ENHANCER_SCOPE_CONFIG_PATH = (
    OUTPUT_ROOT / "enhancer_k562_rc_on_unfreeze_scope_configs.tsv"
)
EXPORT_SUMMARY_PATH = OUTPUT_ROOT / "epoch_history_export_summary.json"

METRIC_COLUMNS = [
    "train_mse",
    "val_mse",
    "train_pearson",
    "val_pearson",
    "train_cod_r2",
    "val_cod_r2",
]

SELECTED_COLUMN_ORDER = [
    "part_slug",
    "fold",
    "epoch",
    *METRIC_COLUMNS,
    "loss_mode",
    "rc_mode",
    "architecture",
    "training_regime",
    "source_head",
    "unfreeze_scope",
    "base_config_id",
    "run_id",
    "history_status",
    "history_source_kind",
    "history_source_path",
    "mse_interpretation",
]

SCOPE_LABELS = {
    "branched_only": "Branch + output",
    "conv3_plus": "Top convolution block + dense head",
    "full": "Full network",
}


def _load_wandb_exporter():
    spec = importlib.util.spec_from_file_location(
        "local_wandb_history_exporter", WANDB_EXPORTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {WANDB_EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _local_wandb_file(run_id: str) -> Path:
    matches = sorted(WANDB_ROOT.glob(f"run-*-{run_id}/run-{run_id}.wandb"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one local W&B binary for run_id={run_id}; found {matches}"
        )
    return matches[0]


def _clean_epoch_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    required = ["epoch", *METRIC_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"{context}: missing history columns {missing}")

    cleaned = frame.copy()
    for column in required:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=required).copy()
    cleaned["epoch"] = cleaned["epoch"].astype(int)
    if cleaned.empty:
        raise RuntimeError(f"{context}: no complete train/validation metric rows")
    if cleaned["epoch"].duplicated().any():
        # Some local W&B binaries contain the same completed epoch payload twice:
        # once when Lightning logs the epoch and once when W&B commits it. Collapse
        # only equivalent metric payloads (allowing one float32 ULP introduced by
        # serialization); divergent duplicates remain a hard error so we never
        # choose between genuinely different trajectories.
        divergent_epochs = []
        for epoch, epoch_rows in cleaned.groupby("epoch", sort=True):
            metric_spread = (
                epoch_rows[METRIC_COLUMNS].max()
                - epoch_rows[METRIC_COLUMNS].min()
            ).abs()
            if len(epoch_rows) > 1 and (metric_spread > 5e-8).any():
                divergent_epochs.append(int(epoch))
        if divergent_epochs:
            raise RuntimeError(
                f"{context}: divergent duplicate complete epoch rows "
                f"{divergent_epochs}"
            )
        ordering_column = (
            "_step"
            if "_step" in cleaned.columns
            else "_source_record_index"
            if "_source_record_index" in cleaned.columns
            else None
        )
        if ordering_column is not None:
            cleaned = cleaned.sort_values(["epoch", ordering_column])
        cleaned = cleaned.drop_duplicates(subset="epoch", keep="last")
    return cleaned.sort_values("epoch").reset_index(drop=True)


def _read_local_wandb_history(
    exporter: Any,
    *,
    run_id: str,
) -> tuple[pd.DataFrame, Path]:
    run_file = _local_wandb_file(run_id)
    metadata, _config, rows, _columns = exporter.read_wandb_file(run_file)
    observed_run_id = str(metadata.get("run_id", ""))
    if observed_run_id != run_id:
        raise RuntimeError(
            f"Local W&B run ID mismatch: requested {run_id}, observed {observed_run_id}"
        )
    return _clean_epoch_frame(pd.DataFrame(rows), context=f"run {run_id}"), run_file


def _selected_stage2_history(
    stage2_histories: pd.DataFrame,
    selected: pd.Series,
) -> pd.DataFrame:
    frame = stage2_histories.loc[
        stage2_histories["part_slug"].eq(selected["part_slug"])
        & stage2_histories["base_config_id"].eq(selected["base_config_id"])
        & stage2_histories["rc_mode"].eq(selected["rc_mode"])
    ].copy()
    if frame.empty:
        raise RuntimeError(
            "No Stage-2 history for selected policy "
            f"{selected['part_slug']} / {selected['base_config_id']} / {selected['rc_mode']}"
        )

    observed_source_heads = (
        frame["source_head"].dropna().astype(str).loc[lambda values: values.ne("")].unique()
        if "source_head" in frame.columns
        else []
    )
    if len(observed_source_heads) > 1:
        raise RuntimeError(
            f"Selected {selected['part_slug']}: multiple source heads "
            f"{observed_source_heads.tolist()}"
        )
    source_head = (
        observed_source_heads[0]
        if len(observed_source_heads) == 1
        else selected.get("source_head", "")
    )

    frames: list[pd.DataFrame] = []
    for fold, fold_frame in frame.groupby("development_fold", sort=True):
        run_ids = fold_frame["resolved_run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) != 1:
            raise RuntimeError(
                f"Selected {selected['part_slug']} fold {fold}: expected one run ID, "
                f"found {run_ids}"
            )
        cleaned = _clean_epoch_frame(
            fold_frame,
            context=f"selected {selected['part_slug']} Stage-2 fold {fold}",
        )
        cleaned["part_slug"] = selected["part_slug"]
        cleaned["fold"] = int(fold)
        cleaned["loss_mode"] = selected["loss_mode"]
        cleaned["rc_mode"] = selected["rc_mode"]
        cleaned["architecture"] = selected["architecture"]
        cleaned["training_regime"] = selected["training_regime"]
        cleaned["source_head"] = source_head
        cleaned["unfreeze_scope"] = selected.get("unfreeze_scope", "")
        cleaned["base_config_id"] = selected["base_config_id"]
        cleaned["run_id"] = run_ids[0]
        cleaned["history_status"] = "exact_selected_policy_history"
        cleaned["history_source_kind"] = "stage2_frozen_local_history_export"
        cleaned["history_source_path"] = str(STAGE2_HISTORIES_PATH.resolve())
        cleaned["mse_interpretation"] = (
            "unweighted MSE on the standardized training target"
        )
        frames.append(cleaned[SELECTED_COLUMN_ORDER])

    result = pd.concat(frames, ignore_index=True)
    _assert_five_folds(result, context=f"selected {selected['part_slug']}")
    return result


def _selected_stage3_history(
    exporter: Any,
    registry: pd.DataFrame,
    selected: pd.Series,
) -> pd.DataFrame:
    mask = (
        registry["campaign_stage"].astype(str).eq("stage3_weighted_loss")
        & registry["part_slug"].astype(str).eq(str(selected["part_slug"]))
        & registry["base_config_id"].astype(str).eq(str(selected["base_config_id"]))
        & registry["rc_mode"].astype(str).eq(str(selected["rc_mode"]))
        & registry["loss_mode"].astype(str).eq(str(selected["loss_mode"]))
    )
    runs = registry.loc[mask].copy()
    if len(runs) != 5:
        raise RuntimeError(
            f"Selected {selected['part_slug']}: expected five Stage-3 registry rows, "
            f"found {len(runs)}"
        )

    runs["development_fold"] = pd.to_numeric(
        runs["development_fold"], errors="raise"
    ).astype(int)
    if set(runs["development_fold"]) != set(range(5)):
        raise RuntimeError(
            f"Selected {selected['part_slug']}: unexpected folds "
            f"{sorted(runs['development_fold'].tolist())}"
        )

    frames: list[pd.DataFrame] = []
    for row in runs.sort_values("development_fold").itertuples(index=False):
        run_id = str(row.run_id)
        history, run_file = _read_local_wandb_history(exporter, run_id=run_id)
        history["part_slug"] = selected["part_slug"]
        history["fold"] = int(row.development_fold)
        history["loss_mode"] = selected["loss_mode"]
        history["rc_mode"] = selected["rc_mode"]
        history["architecture"] = selected["architecture"]
        history["training_regime"] = selected["training_regime"]
        history["source_head"] = selected.get("source_head", "")
        history["unfreeze_scope"] = selected.get("unfreeze_scope", "")
        history["base_config_id"] = selected["base_config_id"]
        history["run_id"] = run_id
        history["history_status"] = "exact_selected_policy_history"
        history["history_source_kind"] = "stage3_local_wandb_binary"
        history["history_source_path"] = str(run_file.resolve())
        history["mse_interpretation"] = (
            "unweighted MSE on the standardized training target; "
            "training objective used barcode weights"
        )
        frames.append(history[SELECTED_COLUMN_ORDER])

    result = pd.concat(frames, ignore_index=True)
    _assert_five_folds(result, context=f"selected {selected['part_slug']}")
    return result


def _assert_five_folds(frame: pd.DataFrame, *, context: str) -> None:
    folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
    if folds != set(range(5)):
        raise RuntimeError(f"{context}: expected folds 0-4, observed {sorted(folds)}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _enhancer_scope_identities(
    scope_rows: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], str]:
    manifests = _read_jsonl(STAGE2_MANIFEST_PATH)
    identities: dict[str, dict[str, Any]] = {}
    for scope_row in scope_rows.itertuples(index=False):
        matches = [
            row
            for row in manifests
            if row.get("base_config_id") == scope_row.base_config_id
            and row.get("source_head") == "K562"
            and row.get("unfreeze_scope") == scope_row.unfreeze_scope
            and row.get("rc_mode") == "on"
        ]
        if not matches:
            raise RuntimeError(
                f"No Stage-2 manifest rows for Enhancer scope {scope_row.unfreeze_scope}"
            )
        identities[scope_row.unfreeze_scope] = dict(matches[0]["base_identity"])

    normalized: dict[str, str] = {}
    for scope, identity in identities.items():
        comparison_identity = dict(identity)
        comparison_identity.pop("unfreeze_scope", None)
        comparison_identity.pop("source_head", None)
        normalized[scope] = json.dumps(
            comparison_identity, sort_keys=True, separators=(",", ":")
        )
    if len(set(normalized.values())) != 1:
        raise RuntimeError(
            "Enhancer K562 RC-on scope rows differ in fields other than unfreeze scope"
        )
    matched_sha256 = hashlib.sha256(
        next(iter(normalized.values())).encode("utf-8")
    ).hexdigest()
    return identities, matched_sha256


def _enhancer_scope_exports(
    stage2_histories: pd.DataFrame,
    stage2_oof: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes = stage2_oof.loc[
        stage2_oof["part_slug"].eq("enhancer")
        & stage2_oof["analysis_lane"].eq("enhancer_transfer_challenger")
        & stage2_oof["source_head"].eq("K562")
        & stage2_oof["rc_mode"].eq("on")
        & stage2_oof["unfreeze_scope"].isin(SCOPE_LABELS)
    ].copy()
    if len(scopes) != 3 or set(scopes["unfreeze_scope"]) != set(SCOPE_LABELS):
        raise RuntimeError(
            "Expected exactly one K562 RC-on Enhancer OOF row for each unfreeze scope"
        )
    if scopes["unfreeze_scope"].duplicated().any():
        raise RuntimeError("Duplicate K562 RC-on Enhancer unfreeze-scope rows")

    _identities, matched_identity_sha256 = _enhancer_scope_identities(scopes)
    history_frames: list[pd.DataFrame] = []
    config_rows: list[dict[str, Any]] = []
    for scope_row in scopes.sort_values(
        "unfreeze_scope",
        key=lambda values: values.map(
            {"branched_only": 0, "conv3_plus": 1, "full": 2}
        ),
    ).itertuples(index=False):
        scope = str(scope_row.unfreeze_scope)
        frame = stage2_histories.loc[
            stage2_histories["part_slug"].eq("enhancer")
            & stage2_histories["analysis_lane"].eq(
                "enhancer_transfer_challenger"
            )
            & stage2_histories["source_head"].eq("K562")
            & stage2_histories["unfreeze_scope"].eq(scope)
            & stage2_histories["rc_mode"].eq("on")
            & stage2_histories["base_config_id"].eq(scope_row.base_config_id)
        ].copy()
        if frame.empty:
            raise RuntimeError(f"No local history rows for Enhancer scope {scope}")

        scope_frames: list[pd.DataFrame] = []
        for fold, fold_frame in frame.groupby("development_fold", sort=True):
            run_ids = (
                fold_frame["resolved_run_id"].dropna().astype(str).unique().tolist()
            )
            if len(run_ids) != 1:
                raise RuntimeError(
                    f"Enhancer {scope} fold {fold}: expected one run ID, found {run_ids}"
                )
            cleaned = _clean_epoch_frame(
                fold_frame, context=f"Enhancer {scope} fold {fold}"
            )
            cleaned["part_slug"] = "enhancer"
            cleaned["fold"] = int(fold)
            cleaned["loss_mode"] = "unweighted_mse"
            cleaned["rc_mode"] = "on"
            cleaned["architecture"] = "BassetBranched"
            cleaned["training_regime"] = "transfer"
            cleaned["source_head"] = "K562"
            cleaned["unfreeze_scope"] = scope
            cleaned["scope_label"] = SCOPE_LABELS[scope]
            cleaned["scope_order"] = {
                "branched_only": 1,
                "conv3_plus": 2,
                "full": 3,
            }[scope]
            cleaned["scope_transition_epoch"] = (
                pd.NA if scope == "branched_only" else 2
            )
            cleaned["base_config_id"] = scope_row.base_config_id
            cleaned["run_id"] = run_ids[0]
            cleaned["history_status"] = "exact_controlled_scope_history"
            cleaned["history_source_kind"] = "stage2_frozen_local_history_export"
            cleaned["history_source_path"] = str(STAGE2_HISTORIES_PATH.resolve())
            cleaned["mse_interpretation"] = (
                "unweighted MSE on the standardized training target"
            )
            cleaned["matched_non_scope_identity_sha256"] = matched_identity_sha256
            scope_frames.append(cleaned)

        scope_history = pd.concat(scope_frames, ignore_index=True)
        _assert_five_folds(scope_history, context=f"Enhancer scope {scope}")
        history_frames.append(scope_history)
        config_rows.append(
            {
                "scope_order": {"branched_only": 1, "conv3_plus": 2, "full": 3}[
                    scope
                ],
                "unfreeze_scope": scope,
                "scope_label": SCOPE_LABELS[scope],
                "base_config_id": scope_row.base_config_id,
                "source_head": "K562",
                "rc_mode": "on",
                "loss_mode": "unweighted_mse",
                "architecture": "BassetBranched",
                "training_regime": "transfer",
                "n_folds": int(scope_history["fold"].nunique()),
                "n_complete_epoch_rows": int(len(scope_history)),
                "minimum_last_epoch_across_folds": int(
                    scope_history.groupby("fold")["epoch"].max().min()
                ),
                "maximum_last_epoch_across_folds": int(
                    scope_history.groupby("fold")["epoch"].max().max()
                ),
                "pooled_oof_pearson": scope_row.pooled_oof_pearson,
                "pooled_oof_rmse": scope_row.pooled_oof_rmse,
                "pooled_oof_cod_r2": scope_row.pooled_oof_cod_r2,
                "scope_transition_epoch": (
                    pd.NA if scope == "branched_only" else 2
                ),
                "matched_non_scope_identity_sha256": matched_identity_sha256,
                "controlled_comparison_status": (
                    "verified_identical_except_unfreeze_scope"
                ),
            }
        )

    history = pd.concat(history_frames, ignore_index=True)
    history_columns = [
        "part_slug",
        "fold",
        "epoch",
        *METRIC_COLUMNS,
        "unfreeze_scope",
        "scope_label",
        "scope_order",
        "scope_transition_epoch",
        "source_head",
        "rc_mode",
        "loss_mode",
        "architecture",
        "training_regime",
        "base_config_id",
        "run_id",
        "history_status",
        "history_source_kind",
        "history_source_path",
        "mse_interpretation",
        "matched_non_scope_identity_sha256",
    ]
    return history[history_columns], pd.DataFrame(config_rows)


def main() -> int:
    for path in [
        SELECTED_POLICIES_PATH,
        STAGE2_HISTORIES_PATH,
        STAGE2_OOF_PATH,
        STAGE2_MANIFEST_PATH,
        RUN_REGISTRY_PATH,
        WANDB_EXPORTER_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    exporter = _load_wandb_exporter()
    selected = pd.read_csv(SELECTED_POLICIES_PATH, keep_default_na=False)
    stage2_histories = pd.read_csv(
        STAGE2_HISTORIES_PATH, sep="\t", low_memory=False
    )
    stage2_oof = pd.read_csv(STAGE2_OOF_PATH)
    registry = pd.read_csv(RUN_REGISTRY_PATH, low_memory=False)

    expected_parts = {"enhancer", "promoter", "intron", "utr3", "utr5"}
    if set(selected["part_slug"]) != expected_parts or len(selected) != 5:
        raise RuntimeError(
            f"Expected one selected policy for each CRE part; observed "
            f"{selected['part_slug'].tolist()}"
        )

    selected_frames: list[pd.DataFrame] = []
    for _, policy in selected.iterrows():
        if policy["loss_mode"] == "barcode_weighted_mse":
            selected_frames.append(
                _selected_stage3_history(exporter, registry, policy)
            )
        elif policy["loss_mode"] == "unweighted_mse":
            selected_frames.append(_selected_stage2_history(stage2_histories, policy))
        else:
            raise RuntimeError(
                f"Unsupported selected loss_mode={policy['loss_mode']!r}"
            )

    selected_history = pd.concat(selected_frames, ignore_index=True)
    selected_history = selected_history.sort_values(
        ["part_slug", "fold", "epoch"]
    ).reset_index(drop=True)
    if selected_history["history_status"].ne(
        "exact_selected_policy_history"
    ).any():
        raise RuntimeError("An unexpected selected-policy history fallback was used")

    enhancer_history, enhancer_configs = _enhancer_scope_exports(
        stage2_histories, stage2_oof
    )
    enhancer_history = enhancer_history.sort_values(
        ["scope_order", "fold", "epoch"]
    ).reset_index(drop=True)
    enhancer_configs = enhancer_configs.sort_values("scope_order").reset_index(
        drop=True
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_history.to_csv(SELECTED_OUTPUT_PATH, sep="\t", index=False)
    enhancer_history.to_csv(ENHANCER_SCOPE_OUTPUT_PATH, sep="\t", index=False)
    enhancer_configs.to_csv(ENHANCER_SCOPE_CONFIG_PATH, sep="\t", index=False)

    selected_counts = (
        selected_history.groupby(
            [
                "part_slug",
                "architecture",
                "loss_mode",
                "rc_mode",
                "history_status",
            ],
            dropna=False,
        )
        .agg(
            n_folds=("fold", "nunique"),
            n_complete_epoch_rows=("epoch", "size"),
            minimum_last_epoch_across_folds=(
                "epoch",
                lambda values: int(
                    selected_history.loc[values.index]
                    .groupby("fold")["epoch"]
                    .max()
                    .min()
                ),
            ),
            maximum_last_epoch_across_folds=(
                "epoch",
                lambda values: int(
                    selected_history.loc[values.index]
                    .groupby("fold")["epoch"]
                    .max()
                    .max()
                ),
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "selected_policy_history_output": str(SELECTED_OUTPUT_PATH.resolve()),
        "selected_policy_complete_epoch_rows": int(len(selected_history)),
        "selected_policy_counts": selected_counts,
        "selected_policy_fallback_count": int(
            selected_history["history_status"].str.contains("fallback").sum()
        ),
        "enhancer_scope_history_output": str(
            ENHANCER_SCOPE_OUTPUT_PATH.resolve()
        ),
        "enhancer_scope_config_output": str(ENHANCER_SCOPE_CONFIG_PATH.resolve()),
        "enhancer_scope_complete_epoch_rows": int(len(enhancer_history)),
        "test_or_final_test_columns_included": False,
        "mse_interpretation": (
            "Canonical train/validation MSE is an unweighted diagnostic on the "
            "standardized training target. Four selected policies optimized a "
            "barcode-weighted training objective."
        ),
    }
    EXPORT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
