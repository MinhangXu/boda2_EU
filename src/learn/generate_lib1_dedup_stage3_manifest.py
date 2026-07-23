#!/usr/bin/env python3
"""Generate the frozen Lib1 dedup Stage 3 weighted-loss manifests.

The launch product contains only the 450 missing barcode-weighted cells.  Its
analysis companion contains those cells plus 450 immutable unweighted OOF
cells reused from Stage 2 or the targeted 3'UTR HPO.  This program reads only
development manifests, development metric tables, and existing validation
prediction/provenance files.  It never imports a DataModule, reads audit IDs,
or constructs an audit/test loader.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_lib1_dedup_stage2_manifest as stage2


CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "stage3_weighted_loss"
MANIFEST_TAG = "lib1_dedup_stage3_weighted_loss_july2026"
MANIFEST_STATUS = "frozen_dry_run_not_launched"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
MODEL_SEED = 1701
FOLDS = tuple(range(5))
PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}
RC_MODES_BY_PART = {
    "enhancer": ("off", "on"),
    "promoter": ("off", "on"),
    "intron": ("off", "on"),
    "utr3": ("off",),
    "utr5": ("off", "on"),
}
EXPECTED_CONFIGS = 50
EXPECTED_WEIGHTED_CELLS = 450
EXPECTED_REUSE_CELLS = 450
EXPECTED_ANALYSIS_CELLS = 900
EXPECTED_OOF_ARMS = 180
WEIGHT_CAP = 8.0
WEIGHT_MIN = 0.1
MARGIN_FRACTION = 0.01
NUMERIC_EPSILON = 1e-12


PORTFOLIOS = {
    "enhancer": (
        "basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036",
        "basecfg_e53d6596a16e9f43bfe71e4ea2a364dd30237733beee9030030ecbc84f6d30a0",
        "basecfg_3f7d963d6d647ee5eb5ee02239f1b0c992c3f33d90200d52b4e00c88e7ddd02d",
        "basecfg_f199d009d69405a41890a39cf91759eeb6c27df03f0082200b4505f78918b82b",
        "basecfg_404c9e99e7e9571266e83c07b5a5016a731b52212ba3723d98a6d0b44b378cec",
        "basecfg_d7ab0bf6f1bc39af9c4ff9269d2ed0e47f5720b1933057b9654a388f5ed0422f",
        "basecfg_5d9f63c25515a73921372a308950d2d79367da2c06e35899575ff2b88c000b5e",
        "basecfg_18119f07e851868812804e4fd3e36585fa0e472b47e71c913886e7ebba668bd9",
        "basecfg_7bb5763f52f3678922d64e5026e75fa14b79bde606319b207a5f8b30885f87b8",
        "basecfg_246106d4d9907232c48b9d670cb58642ec84b1ac712d4ce21636ff0d33a81c18",
    ),
    "promoter": (
        "basecfg_00175f1ce3e6b9bb7d49b89360083a7314cb294edf173d05d9f076913387c74e",
        "basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d",
        "basecfg_e10d0e2bdadc81888c0cd24f22194f01c9bb752fb97bcad66b6fb20da5fe66eb",
        "basecfg_0c0cefe749c9241f03f893c1fcfe585418a91d151686ee2d4b0eca54335790f8",
        "basecfg_9b9293193ecdac4bffee9b00e58cfdde742789ac1c2d1d625047d4578e4fc5fe",
        "basecfg_f3fa8318ff61c1cb8758134e7dbb9ad2640bdb358c2a736c5d96495080105c4d",
        "basecfg_9821907e1ab3069b1657e66e9befa92e967038385a0909eb1bda10b1d2df24d0",
        "basecfg_badc6370f710b1ac55fcd2d4d6de22daa862d46aac6c273fa25fdb638bb8f46c",
        "basecfg_fe3d6b7e556cd3237d8f537038331229d2a708f32de8d5391056bb5b02ac16f0",
        "basecfg_408bbe2f201458b3b2c75f768501e9cc824bbcc62f120911829322422bea82cb",
    ),
    "intron": (
        "basecfg_6079cd38f32d3f5cf024c66fb43e7f88c2ced932f984fbebe30ba99672641b74",
        "basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86",
        "basecfg_0ee9e54c8bfb2917566afaa790fcd981a007bb4c35d8427b5a83ba69335c08f3",
        "basecfg_873605b1a4643a9a8745b10c68faf5f3d485637b9677805410b172f71af146f1",
        "basecfg_767a6d28b3510037a8510a0d41e00df9106064b13af76975e225bd0e8bcb94d7",
        "basecfg_710db0cc09f3c386b726a49fd23be30e0fea4896711404a393d54b60d945de4f",
        "basecfg_a76fef1421c97368714a0ae354db301f69a9a6a7561244523b4106a65fe4a093",
        "basecfg_5b5d2d82cef98c6e0c7522dbbc388ef4da59ee65687f40159e7c9548eb2277f3",
        "basecfg_0c59aba4ea114b651fae8352b2a9a3f9010edbd17fe288b902d80ba25c2a0223",
        "basecfg_e3b7fc22d2bedc66c15d3e7ce8aaaa44679a8305e1719a85c3f8bb51dcb508ca",
    ),
    "utr3": (
        "basecfg_6cb459958ae1a16e112bdacc6e03c9e02fc12cdc85ed951cfcd25ada7856a517",
        "basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe",
        "basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062",
        "basecfg_8b14e9e7f2f26e52985dda2dec8f128c9da9a31662a64015dca76a993b4cd5b4",
        "basecfg_1becdea28bb6a22dbb61a48222baf1cbce413ac6e405691c9bda4b1da6253f90",
        "basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05",
        "basecfg_1e3a0c9f053271a63a4da596c588484b52c56cf65fe6fb791bd909e15c3b9def",
        "basecfg_ec031204c44d76ed859477d8b2fcb74f54daf5a9d6d70017728dac5dcabbeb2b",
        "basecfg_585fba9a4fec47048843b484fad428e9a5236fffe3aec370c3938fb4db39fa92",
        "basecfg_231fe76767cea395f9dc5ae2625155780ac85b83d944e3ce97ba494417a21fd7",
    ),
    "utr5": (
        "basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315",
        "basecfg_25d3b0fb122d4da050145825875c04f5cedc047178b5d2d159d2275a5731f227",
        "basecfg_e3b85c86fe400906280db9093b388bb1b74a552467120eac98e86c5202650d17",
        "basecfg_99b40ac8bca80e76b56403be8b15214c10cf6fc33730d7dd3926997792fef16b",
        "basecfg_bee0f2b508e0fbc529890aafd7b63c93a4014e7bee8ecd46f99e9ddb5481be5f",
        "basecfg_c9a37b4a162fd8fefbde5b01aaf7556931ec254ec2f1abefa2c9b0f4becb4b56",
        "basecfg_ffd4992641df6d33f2b23c1aa5857ceab29a6ae247d489d220b53177871f1369",
        "basecfg_65e011f225d06cf57c83ac305545a839271748462292e326ce22262b13c5fe94",
        "basecfg_2106736d06b1570dbc9725701e675122292ada6893680b141ece9a9c7a79e82b",
        "basecfg_d5ad87bb22a68b1d8dd7d91351fafcb8f2d38ac7b7d3f40bc17947d4b9a28be8",
    ),
}


def portfolio_role(part: str, rank: int) -> str:
    if part == "enhancer":
        return "complete_transfer_grid" if rank <= 6 else f"scratch_anchor_{rank - 6}"
    if part == "utr3":
        roles = {
            1: "utrbasset_numerical_winner",
            2: "utrbasset_stage2_incumbent",
            3: "utrbasset_performance_candidate",
            4: "utrbasset_performance_candidate",
            5: "utrbasset_performance_candidate",
            6: "utrbasset_calibration_candidate",
            7: "utrbasset_worst_fold_robustness_candidate",
            8: "resnet_architecture_anchor_1",
            9: "resnet_architecture_anchor_2",
            10: "resnet_architecture_anchor_3",
        }
        return roles[rank]
    if part == "utr5" and rank in (7, 8):
        return f"resnet_architecture_anchor_{rank - 6}"
    return "completed_stage2_nominee"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(stage2.canonical_json(row) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wandb_project(part: str) -> str:
    return f"{part}__bashor_in_house__dedup_exact_v1__stage3_weighted_development"


def make_ids(part: str, base_config_id: str, fold: int, rc_mode: str, loss_mode: str):
    shared = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "part_slug": part,
        "base_config_id": base_config_id,
        "development_fold": int(fold),
        "rc_mode": rc_mode,
        "model_seed": MODEL_SEED,
    }
    loss_pair_id = "losspair_" + stage2.sha256_json(shared)[:20]
    rc_payload = dict(shared)
    rc_payload.pop("rc_mode")
    rc_payload["loss_mode"] = loss_mode
    rc_pair_id = (
        "" if part == "utr3" else "rcpair_" + stage2.sha256_json(rc_payload)[:20]
    )
    cell_payload = dict(shared)
    cell_payload["loss_mode"] = loss_mode
    cell_id = "cell_" + stage2.sha256_json(cell_payload)[:20]
    return loss_pair_id, rc_pair_id, cell_id


def prediction_evidence(source: dict) -> dict:
    prediction_dir = Path(source["default_root_dir"]) / "predictions"
    candidates = sorted(prediction_dir.glob("*__val_predictions.tsv*"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one validation prediction under {prediction_dir}; found {len(candidates)}"
        )
    prediction = candidates[0]
    run_id = prediction.name.split("__", 1)[0]
    provenance = Path(source["default_root_dir"]) / "provenance" / f"{run_id}__run_provenance.json"
    if not provenance.is_file():
        raise FileNotFoundError(provenance)
    payload = json.loads(provenance.read_text())
    split_summary = payload.get("data_split_summary", {})
    if split_summary.get("n_test") != 0:
        raise ValueError(f"Source cell {source['cell_id']} contains test/audit evaluation")
    with prediction.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        n_rows = sum(1 for line in handle if line.strip())
    required = {"construct_id", "log2_RNA_DNA", "prediction_raw"}
    if not required.issubset(header) or n_rows <= 0:
        raise ValueError(f"Invalid source validation prediction {prediction}")
    return {
        "source_run_id": run_id,
        "source_prediction_path": str(prediction.resolve()),
        "source_prediction_sha256": stage2.sha256_file(prediction),
        "source_prediction_rows": n_rows,
        "source_provenance_path": str(provenance.resolve()),
        "source_provenance_sha256": stage2.sha256_file(provenance),
        "source_val_row_id_hash": split_summary.get("val_row_id_hash", ""),
    }


def source_index(stage2_path: Path, targeted_path: Path):
    manifests = (
        ("stage2", stage2_path, read_jsonl(stage2_path)),
        ("targeted_utr3", targeted_path, read_jsonl(targeted_path)),
    )
    index = {}
    templates = {}
    source_hashes = {}
    for label, path, rows in manifests:
        source_hashes[label] = stage2.sha256_file(path)
        for row in rows:
            key = (
                row["part_slug"],
                row["base_config_id"],
                int(row["development_fold"]),
                row["rc_mode"],
            )
            if key in index:
                raise ValueError(f"Duplicate source condition across manifests: {key}")
            wrapped = copy.deepcopy(row)
            wrapped["_source_manifest_label"] = label
            wrapped["_source_manifest_path"] = str(path.resolve())
            wrapped["_source_manifest_sha256"] = source_hashes[label]
            index[key] = wrapped
            if row.get("train_command"):
                templates.setdefault((label, row["base_config_id"]), row["train_command"])
    return index, templates, source_hashes


def common_row(
    source: dict,
    evidence: dict,
    *,
    rank: int,
    role: str,
    loss_mode: str,
) -> dict:
    part = source["part_slug"]
    fold = int(source["development_fold"])
    rc_mode = source["rc_mode"]
    loss_pair_id, rc_pair_id, cell_id = make_ids(
        part, source["base_config_id"], fold, rc_mode, loss_mode
    )
    return {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "manifest_tag": MANIFEST_TAG,
        "manifest_status": MANIFEST_STATUS,
        "design_type": (
            "rc_off_matched_loss" if part == "utr3" else "rc_by_loss_factorial"
        ),
        "part_slug": part,
        "portfolio_rank": rank,
        "portfolio_role": role,
        "portfolio_eligible_for_final_selection": True,
        "analysis_lane": source["analysis_lane"],
        "challenger_family": source.get("challenger_family", ""),
        "config_origin": source.get("config_origin", ""),
        "training_regime": source["training_regime"],
        "architecture": source["architecture"],
        "architecture_slug": source.get("architecture_slug", ""),
        "base_config_id": source["base_config_id"],
        "base_config_sha256": source["base_config_sha256"],
        "base_identity": source["base_identity"],
        "policy_id": source.get("policy_id", source["base_config_id"]),
        "initialization": source.get("initialization", "scratch"),
        "source_head": source.get("source_head", ""),
        "source_head_index": source.get("source_head_index"),
        "unfreeze_scope": source.get("unfreeze_scope", ""),
        "input_policy": source.get("input_policy", ""),
        "pretrained_artifact_sha256": source.get("pretrained_artifact_sha256", ""),
        "data_generation_id": source["data_generation_id"],
        "dataset_path": source["dataset_path"],
        "dataset_sha256": source["dataset_sha256"],
        "split_manifest_id": source["split_manifest_id"],
        "split_manifest_path": source["split_manifest_path"],
        "split_manifest_sha256": source["split_manifest_sha256"],
        "development_fold": fold,
        "model_seed": MODEL_SEED,
        "use_reverse_complements": rc_mode == "on",
        "rc_mode": rc_mode,
        "loss_mode": loss_mode,
        "loss_pair_id": loss_pair_id,
        "rc_pair_id": rc_pair_id,
        "cell_id": cell_id,
        "target_column": "log2_RNA_DNA",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "length_policy": source.get("length_policy", ""),
        "barcode_weight_cap": WEIGHT_CAP,
        "barcode_weight_min": WEIGHT_MIN,
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "source_manifest_label": source["_source_manifest_label"],
        "source_manifest_path": source["_source_manifest_path"],
        "source_manifest_sha256": source["_source_manifest_sha256"],
        "source_manifest_row": source.get("analysis_cell", source.get("manifest_row")),
        "source_unweighted_cell_id": source["cell_id"],
        "source_unweighted_row_fingerprint": source["row_fingerprint"],
        "source_unweighted_rc_pair_id": source.get("rc_pair_id", ""),
        **evidence,
    }


def reuse_row(source: dict, evidence: dict, rank: int, role: str) -> dict:
    row = common_row(
        source, evidence, rank=rank, role=role, loss_mode="unweighted_mse"
    )
    row.update(
        {
            "execution_disposition": "reuse_unweighted",
            "barcode_weighting": False,
            "graph_module": source["base_identity"].get("graph_module", ""),
            "logger_project": source["logger_project"],
            "wandb_group": source.get("wandb_group", ""),
            "wandb_job_type": source.get("wandb_job_type", ""),
            "planned_run_name": source["planned_run_name"],
            "default_root_dir": source["default_root_dir"],
            "train_command": "",
        }
    )
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def weighted_row(
    source: dict,
    evidence: dict,
    template_command: str,
    rank: int,
    role: str,
    output_root: Path,
) -> dict:
    row = common_row(
        source, evidence, rank=rank, role=role, loss_mode="barcode_weighted_mse"
    )
    part = row["part_slug"]
    prefix = row["base_config_sha256"][:16]
    root = (
        output_root
        / part
        / row["base_config_id"]
        / f"fold_{row['development_fold']}"
        / f"rc_{row['rc_mode']}"
        / "barcode_weighted_mse"
    )
    project = wandb_project(part)
    group = f"{CAMPAIGN_ID}__stage3__{part}__{row['analysis_lane']}"
    run_name = (
        f"{MANIFEST_TAG}__{part}__p{rank:02d}__{prefix}__"
        f"fold{row['development_fold']}__rc_{row['rc_mode']}__weighted"
    )
    graph_module = (
        "CNNBassetBranchedScopedWeightedTransfer"
        if row["training_regime"] == "transfer"
        else "CNNWeightedRegressionTraining"
    )

    options = stage2.parse_command(template_command)
    replacements = {
        "graph_module": graph_module,
        "artifact_path": str(root / "artifacts"),
        "best_checkpoint_dir": str(root / "published_checkpoint_disabled"),
        "prediction_output_dir": str(root / "predictions"),
        "provenance_output_dir": str(root / "provenance"),
        "default_root_dir": str(root),
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "part_slug": part,
        "analysis_lane": row["analysis_lane"],
        "challenger_family": row["challenger_family"],
        "policy_id": row["policy_id"],
        "config_origin": row["config_origin"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "rc_pair_id": row["rc_pair_id"],
        "loss_pair_id": row["loss_pair_id"],
        "source_unweighted_cell_id": row["source_unweighted_cell_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "initialization": row["initialization"],
        "input_policy": row["input_policy"],
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": row["development_fold"],
        "split_fold": row["development_fold"],
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": MODEL_SEED,
        "loss_mode": "barcode_weighted_mse",
        "target_definition": row["target_definition"],
        "length_policy": row["length_policy"],
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "use_reverse_complements": row["rc_mode"] == "on",
        "barcode_weighting": True,
        "barcode_weight_cap": WEIGHT_CAP,
        "barcode_weight_min": WEIGHT_MIN,
        "loss_criterion": "MSELoss",
        "reduction": "mean",
        "logger_type": "wandb",
        "logger_project": project,
        "wandb_entity": EXPECTED_ENTITY,
        "wandb_group": group,
        "wandb_job_type": "stage3_weighted_cell",
        "run_name": run_name,
        "exact_run_name": True,
        "enable_progress_bar": False,
    }
    for name, value in replacements.items():
        stage2.put(options, name, value)
    stage2.put(
        options,
        "wandb_tags",
        [
            CAMPAIGN_ID,
            CAMPAIGN_STAGE,
            part,
            row["analysis_lane"],
            row["architecture_slug"],
            f"portfolio_rank_{rank:02d}",
            role,
            f"fold{row['development_fold']}",
            f"rc_{row['rc_mode']}",
            "barcode_weighted_mse",
            "seed1701",
        ],
    )
    if row["training_regime"] == "transfer":
        options.pop("weighted_loss_reduction", None)
    else:
        stage2.put(options, "weighted_loss_reduction", "mean")

    row.update(
        {
            "execution_disposition": "launch",
            "barcode_weighting": True,
            "graph_module": graph_module,
            "logger_project": project,
            "wandb_entity": EXPECTED_ENTITY,
            "wandb_group": group,
            "wandb_job_type": "stage3_weighted_cell",
            "planned_run_name": run_name,
            "default_root_dir": str(root),
            "train_command": stage2.command_from_options(options),
        }
    )
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def row_fingerprint(row: dict) -> str:
    fields = (
        "campaign_id",
        "campaign_stage",
        "part_slug",
        "portfolio_rank",
        "base_config_id",
        "development_fold",
        "rc_mode",
        "loss_mode",
        "loss_pair_id",
        "rc_pair_id",
        "cell_id",
        "execution_disposition",
        "dataset_sha256",
        "split_manifest_sha256",
        "source_unweighted_cell_id",
        "source_prediction_sha256",
        "planned_run_name",
        "train_command",
    )
    return stage2.sha256_json({field: row.get(field) for field in fields})


def metric_margins(
    stage2_metrics_path: Path, targeted_metrics_path: Path
) -> dict[str, dict]:
    stage2_rows = read_csv(stage2_metrics_path)
    targeted_rows = read_csv(targeted_metrics_path)
    margins = {}
    for part, config_ids in PORTFOLIOS.items():
        source = targeted_rows if part == "utr3" else stage2_rows
        selected = [
            row
            for row in source
            if row["base_config_id"] in config_ids
            and row.get("part_slug", part) == part
            and (part != "utr3" or row["rc_mode"] == "off")
        ]
        expected = 10 if part == "utr3" else 20
        if len(selected) != expected or len({row["base_config_id"] for row in selected}) != 10:
            raise ValueError(f"{part}: expected {expected} frozen metric arms; found {len(selected)}")
        rmses = [float(row["pooled_oof_rmse"]) for row in selected]
        cods = [float(row["pooled_oof_cod_r2"]) for row in selected]
        variances = [rmse * rmse / (1.0 - cod) for rmse, cod in zip(rmses, cods)]
        if max(variances) - min(variances) > 1e-10:
            raise ValueError(f"{part}: OOF target variance is not invariant across arms")
        reference_rmse = float(statistics.median(rmses))
        target_variance = float(statistics.median(variances))
        rmse_margin = MARGIN_FRACTION * reference_rmse
        cod_margin = (
            (reference_rmse + rmse_margin) ** 2 - reference_rmse ** 2
        ) / target_variance
        margins[part] = {
            "source_unweighted_arm_count": expected,
            "reference_median_pooled_rmse": reference_rmse,
            "reference_oof_target_variance": target_variance,
            "relative_rmse_fraction": MARGIN_FRACTION,
            "allowed_pooled_rmse_increase": rmse_margin,
            "allowed_pooled_cod_r2_decrease": cod_margin,
            "numeric_epsilon": NUMERIC_EPSILON,
        }
    return margins


def portfolio_artifact(index: dict, margins: dict, source_hashes: dict, args) -> dict:
    rows = []
    for part in sorted(PORTFOLIOS, key=PART_ORDER.get):
        for rank, base_config_id in enumerate(PORTFOLIOS[part], 1):
            rc_mode = RC_MODES_BY_PART[part][0]
            source = index[(part, base_config_id, 0, rc_mode)]
            rows.append(
                {
                    "part_slug": part,
                    "portfolio_rank": rank,
                    "portfolio_role": portfolio_role(part, rank),
                    "eligible_for_final_selection": True,
                    "base_config_id": base_config_id,
                    "architecture": source["architecture"],
                    "analysis_lane": source["analysis_lane"],
                    "training_regime": source["training_regime"],
                    "source_head": source.get("source_head", ""),
                    "unfreeze_scope": source.get("unfreeze_scope", ""),
                    "rc_modes": list(RC_MODES_BY_PART[part]),
                }
            )
    return {
        "schema_version": "lib1_dedup_stage3_portfolio_v1",
        "protocol_date": "2026-07-14",
        "manifest_status": MANIFEST_STATUS,
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "portfolio_policy": {
            "configurations_per_part": 10,
            "enhancer": "complete two-head by three-scope transfer grid plus four strongest Stage-2 scratch anchors",
            "promoter": "all ten completed Stage-2 nominees",
            "intron": "all ten completed Stage-2 nominees with inferred-mask robustness reporting",
            "utr3": "seven UTRBasset performance/robustness candidates plus three RC-off ResNet architecture anchors",
            "utr5": "all ten completed Stage-2 nominees, including two ResNet architecture anchors",
            "all_candidates_eligible_for_final_selection": True,
        },
        "weight_policy": {
            "formula": "clip(log1p(n_barcodes)/log1p(8), 0.1, 1.0)",
            "cap": WEIGHT_CAP,
            "minimum": WEIGHT_MIN,
            "training_loss": "sum_i(w_i * mean_j(error_ij^2)) / sum_i(w_i)",
            "training_weights_required": True,
            "validation_metrics_unweighted": True,
        },
        "selection_policy": {
            "primary_metric": "pooled five-fold raw-scale OOF Pearson",
            "admissibility": {
                "rc_off_unweighted": "eligible when all five development folds are complete and every required selection metric is finite",
                "weighted": "eligible only after passing the loss gate against its exact unweighted mate",
                "rc_on": "eligible only after passing the RC gate against the same-loss RC-off arm",
                "rc_on_weighted": "must pass both the loss gate and the RC gate",
            },
            "non_finite_policy": {
                "intervention_gate": "any non-finite required pooled or five-fold gate input makes that gate fail",
                "arm_selection": "any non-finite pooled Pearson/RMSE/COD or minimum-fold Pearson makes the arm ineligible; Intron additionally requires finite minimum-stratum and within-stratum-centered Pearson",
                "bootstrap": "any non-finite best-arm bootstrap replicate is a fatal analysis-contract error",
                "descriptive_calibration": "report undefined with reason; descriptive calibration alone does not determine admissibility",
            },
            "loss_gate": {
                "mean_fold_pearson_delta_at_least": 0.005,
                "positive_fold_count_at_least": 4,
                "rmse_and_cod_margins": "part_specific_values_below",
            },
            "rc_gate_for_non_utr3_parts": {
                "mean_fold_pearson_delta_at_least": 0.005,
                "positive_fold_count_at_least": 4,
                "rmse_and_cod_margins": "part_specific_values_below",
            },
            "descriptive_rc_by_loss_interaction": {
                "formula": "(weighted_rc_on-weighted_rc_off)-(unweighted_rc_on-unweighted_rc_off)",
                "complete_factorials": 40,
                "enters_gate_or_selection": False,
            },
            "intron_extra_gate": {
                "applies_to": ["loss_gate", "rc_gate"],
                "mean_within_inferred_stratum_centered_pearson_delta_at_least": 0.0,
                "negative_fold_count_at_most": 2,
            },
            "one_se_definition": {
                "best_arm": "admissible arm with highest pooled five-fold raw-scale OOF Pearson",
                "bootstrap": "10000 within-fold row resamples, concatenate five resampled held-out folds, recompute pooled Pearson",
                "seed": 20260714,
                "rng_scope": "reinitialize numpy.random.default_rng with the frozen seed independently for each part",
                "standard_error": "sample standard deviation of the best arm bootstrap Pearson replicates",
                "band": "candidate point-estimate pooled Pearson >= best point estimate minus best-arm standard error",
                "exact_best_point_tie": "choose bootstrap reference with the same downstream deterministic ordering",
            },
            "final_arm_order": [
                "admissible under complete-OOF and intervention gates",
                "inside best-arm one-SE bootstrap band (10000 fold-stratified resamples; seed 20260714)",
                "highest minimum fold Pearson",
                "Intron only: highest minimum inferred-stratum Pearson, then highest within-inferred-stratum-centered pooled Pearson",
                "lowest pooled RMSE",
                "highest pooled COD R2",
                "exact metric-tie block containing only Enhancer transfer routes: narrower scope branched_only then conv3_plus then full",
                "exact metric-tie block containing any scratch route, or any other part: fewer total parameters",
                "residual equal-parameter Enhancer transfer tie: narrower scope",
                "RC off, then unweighted loss, then lexicographic full base_config_id",
            ],
        },
        "metric_margins": margins,
        "intron_reporting_contract": {
            "labels": ["mask1_specific", "mask2_not_mask1", "mask3_residual"],
            "labels_are": "inferred sequence-mask sensitivity strata, not verified sublibraries",
            "development_oof_n": 1061,
            "development_counts": [374, 365, 322],
            "fold_trained_stratum_mean_baseline_prediction_sha256": "82c228a3ba0cd0b0df403b52095f8efc1a9a3cdd20417a656b8cccb8f2d14e8c",
            "mandatory": [
                "natural-mixture pooled metrics",
                "within-inferred-stratum-centered metrics",
                "macro-stratum and minimum-stratum metrics",
                "all per-stratum raw-scale metrics and calibration",
                "fold-training-fitted stratum-mean baseline",
                "fixed equal-stratum sensitivity with weight range and effective sample size",
            ],
            "final_audit_later": "natural 265-row audit remains primary; same frozen predictions receive sensitivity reporting",
            "audit_category_counts_inspected_now": False,
        },
        "source_manifest_sha256": source_hashes,
        "source_metric_sha256": {
            "stage2": stage2.sha256_file(args.stage2_metrics),
            "targeted_utr3": stage2.sha256_file(args.targeted_metrics),
        },
        "configs": rows,
    }


def validate_rows(analysis_rows: list[dict], launch_rows: list[dict]) -> None:
    if len(analysis_rows) != EXPECTED_ANALYSIS_CELLS:
        raise AssertionError(f"Expected 900 analysis cells; found {len(analysis_rows)}")
    if len(launch_rows) != EXPECTED_WEIGHTED_CELLS:
        raise AssertionError(f"Expected 450 weighted cells; found {len(launch_rows)}")
    if Counter(row["execution_disposition"] for row in analysis_rows) != Counter(
        {"reuse_unweighted": EXPECTED_REUSE_CELLS, "launch": EXPECTED_WEIGHTED_CELLS}
    ):
        raise AssertionError("Stage 3 launch/reuse accounting changed")
    if len({row["cell_id"] for row in analysis_rows}) != len(analysis_rows):
        raise AssertionError("Stage 3 cell_id values are not unique")
    if len({row["row_fingerprint"] for row in analysis_rows}) != len(analysis_rows):
        raise AssertionError("Stage 3 row fingerprints are not unique")
    if len({row["base_config_id"] for row in analysis_rows}) != EXPECTED_CONFIGS:
        raise AssertionError("Stage 3 does not contain exactly 50 configs")

    loss_pairs = defaultdict(list)
    rc_pairs = defaultdict(list)
    arm_keys = set()
    for row in analysis_rows:
        loss_pairs[row["loss_pair_id"]].append(row)
        if row["rc_pair_id"]:
            rc_pairs[row["rc_pair_id"]].append(row)
        arm_keys.add(
            (
                row["part_slug"], row["base_config_id"], row["rc_mode"], row["loss_mode"]
            )
        )
        if row["part_slug"] == "utr3" and row["rc_mode"] != "off":
            raise AssertionError("3'UTR Stage 3 must remain RC-off-only")
        if row["evaluate_test_after_fit"] is not False:
            raise AssertionError("Stage 3 cannot evaluate audit/test data")
        if row["epoch_eval_splits"] != ["train", "val"] or row["prediction_splits"] != ["val"]:
            raise AssertionError("Stage 3 must remain train/val-only")
        if "audit_ids" in row.get("train_command", "").lower():
            raise AssertionError("Stage 3 command contains audit material")
    if len(arm_keys) != EXPECTED_OOF_ARMS:
        raise AssertionError(f"Expected 180 Stage 3 OOF arms; found {len(arm_keys)}")
    if len(loss_pairs) != EXPECTED_WEIGHTED_CELLS:
        raise AssertionError("Expected 450 fold-level loss pairs")
    for pair_id, pair in loss_pairs.items():
        if len(pair) != 2 or {row["loss_mode"] for row in pair} != {
            "unweighted_mse", "barcode_weighted_mse"
        }:
            raise AssertionError(f"Incomplete loss pair {pair_id}")
        invariant = (
            "part_slug", "base_config_id", "development_fold", "rc_mode",
            "model_seed", "dataset_sha256", "split_manifest_sha256",
            "source_unweighted_cell_id", "source_prediction_sha256",
        )
        if any(pair[0][field] != pair[1][field] for field in invariant):
            raise AssertionError(f"Loss-pair invariant mismatch for {pair_id}")
    if len(rc_pairs) != 400:
        raise AssertionError(f"Expected 400 fold-level RC pairs; found {len(rc_pairs)}")
    for pair_id, pair in rc_pairs.items():
        if len(pair) != 2 or {row["rc_mode"] for row in pair} != {"off", "on"}:
            raise AssertionError(f"Incomplete RC pair {pair_id}")

    for row in launch_rows:
        expected_graph = (
            "CNNBassetBranchedScopedWeightedTransfer"
            if row["training_regime"] == "transfer"
            else "CNNWeightedRegressionTraining"
        )
        if row["graph_module"] != expected_graph or row["barcode_weighting"] is not True:
            raise AssertionError(f"Cell {row['cell_id']} is not strictly weighted")
        options = stage2.parse_command(row["train_command"])
        if options.get("epoch_eval_splits") != ["train", "val"]:
            raise AssertionError("Launch command exposes a non-development split")
        if options.get("prediction_splits") != ["val"]:
            raise AssertionError("Launch command exports a non-validation split")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-analysis-manifest",
        type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl",
    )
    parser.add_argument(
        "--targeted-utr3-manifest",
        type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026__dry_run_manifest.jsonl",
    )
    parser.add_argument(
        "--stage2-metrics",
        type=Path,
        default=HERE / "outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv",
    )
    parser.add_argument(
        "--targeted-metrics",
        type=Path,
        default=HERE / "outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/utr3_targeted_hpo_combined_arm_metrics.csv",
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "outputs/hpo_manifests")
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest_tag != MANIFEST_TAG:
        raise ValueError(f"Frozen Stage 3 tag must be {MANIFEST_TAG!r}")
    index, templates, source_hashes = source_index(
        args.stage2_analysis_manifest, args.targeted_utr3_manifest
    )
    margins = metric_margins(args.stage2_metrics, args.targeted_metrics)
    portfolio = portfolio_artifact(index, margins, source_hashes, args)

    output_root = HERE / "outputs/hpo_runs" / MANIFEST_TAG
    analysis_rows = []
    evidence_cache = {}
    for part in sorted(PORTFOLIOS, key=PART_ORDER.get):
        for rank, base_config_id in enumerate(PORTFOLIOS[part], 1):
            role = portfolio_role(part, rank)
            for fold in FOLDS:
                for rc_mode in RC_MODES_BY_PART[part]:
                    key = (part, base_config_id, fold, rc_mode)
                    source = index.get(key)
                    if source is None:
                        raise ValueError(f"Missing immutable unweighted source cell: {key}")
                    evidence_cache.setdefault(source["cell_id"], prediction_evidence(source))
                    evidence = evidence_cache[source["cell_id"]]
                    template_key = (source["_source_manifest_label"], base_config_id)
                    template = templates.get(template_key)
                    if not template:
                        raise ValueError(f"Missing command template for {template_key}")
                    analysis_rows.append(reuse_row(source, evidence, rank, role))
                    analysis_rows.append(
                        weighted_row(source, evidence, template, rank, role, output_root)
                    )

    analysis_rows.sort(
        key=lambda row: (
            PART_ORDER[row["part_slug"]],
            row["portfolio_rank"],
            row["development_fold"],
            0 if row["rc_mode"] == "off" else 1,
            0 if row["loss_mode"] == "unweighted_mse" else 1,
        )
    )
    for index_value, row in enumerate(analysis_rows, 1):
        row["analysis_cell"] = index_value
    launch_rows = [copy.deepcopy(row) for row in analysis_rows if row["execution_disposition"] == "launch"]
    for index_value, row in enumerate(launch_rows, 1):
        row["manifest_row"] = index_value
    reuse_rows = [row for row in analysis_rows if row["execution_disposition"] == "reuse_unweighted"]
    validate_rows(analysis_rows, launch_rows)

    prefix = args.outdir / MANIFEST_TAG
    portfolio_path = Path(str(prefix) + "__portfolio.json")
    analysis_path = Path(str(prefix) + "__analysis_manifest.jsonl")
    launch_path = Path(str(prefix) + "__dry_run_manifest.jsonl")
    reuse_path = Path(str(prefix) + "__unweighted_reuse.jsonl")
    write_json(portfolio_path, portfolio)
    write_jsonl(analysis_path, analysis_rows)
    write_jsonl(launch_path, launch_rows)
    write_jsonl(reuse_path, reuse_rows)
    write_csv(
        Path(str(prefix) + "__dry_run_manifest.csv"),
        launch_rows,
        (
            "manifest_row", "analysis_cell", "cell_id", "loss_pair_id", "rc_pair_id",
            "part_slug", "portfolio_rank", "portfolio_role", "architecture",
            "training_regime", "base_config_id", "development_fold", "rc_mode",
            "source_unweighted_cell_id", "planned_run_name", "logger_project",
            "wandb_group", "row_fingerprint", "train_command",
        ),
    )

    summary = {
        "schema_version": "lib1_dedup_stage3_manifest_v1",
        "protocol_date": "2026-07-14",
        "manifest_status": MANIFEST_STATUS,
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "configs_per_part": 10,
        "total_configs": EXPECTED_CONFIGS,
        "new_weighted_cells": len(launch_rows),
        "unweighted_reuse_cells": len(reuse_rows),
        "analysis_cells": len(analysis_rows),
        "complete_oof_arms": EXPECTED_OOF_ARMS,
        "fold_level_loss_pairs": EXPECTED_WEIGHTED_CELLS,
        "fold_level_rc_pairs": 400,
        "counts_by_part_new_weighted": dict(Counter(row["part_slug"] for row in launch_rows)),
        "rc_modes_by_part": {part: list(values) for part, values in RC_MODES_BY_PART.items()},
        "weight_policy": portfolio["weight_policy"],
        "metric_margins": margins,
        "wandb_entity": EXPECTED_ENTITY,
        "wandb_projects": {part: wandb_project(part) for part in PORTFOLIOS},
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "audit_stratum_counts_inspected": False,
        "commands_executed": 0,
        "portfolio_path": str(portfolio_path.resolve()),
        "portfolio_sha256": stage2.sha256_file(portfolio_path),
        "analysis_manifest_path": str(analysis_path.resolve()),
        "analysis_manifest_sha256": stage2.sha256_file(analysis_path),
        "dry_run_manifest_path": str(launch_path.resolve()),
        "dry_run_manifest_sha256": stage2.sha256_file(launch_path),
        "unweighted_reuse_path": str(reuse_path.resolve()),
        "unweighted_reuse_sha256": stage2.sha256_file(reuse_path),
        "source_manifest_sha256": source_hashes,
        "source_metric_sha256": portfolio["source_metric_sha256"],
    }
    summary_path = Path(str(prefix) + "__summary.json")
    write_json(summary_path, summary)

    print("Generated frozen Lib1 dedup Stage 3 weighted-loss dry run")
    print(f"  portfolios: 5 parts x 10 configs = {EXPECTED_CONFIGS}")
    print(f"  new weighted cells: {len(launch_rows)}")
    print(f"  unweighted reuse cells: {len(reuse_rows)}")
    print(f"  matched analysis cells: {len(analysis_rows)}")
    print(f"  dry-run manifest: {launch_path}")
    print(f"  dry-run SHA256: {summary['dry_run_manifest_sha256']}")
    print("  audit loader instantiated: false")
    print("  commands executed: 0")


if __name__ == "__main__":
    main()
