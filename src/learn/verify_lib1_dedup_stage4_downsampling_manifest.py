#!/usr/bin/env python3
"""Independently verify the frozen Lib1 dedup Stage 4 dry-run manifest.

This verifier does not import the generator or a DataModule. It independently
reconstructs the inner/outer folds and seeded stable-ID prefixes from the
frozen split assignments, checks every command and fingerprint, and confirms
that no final-test evaluation surface is exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_stage4_downsampling_july2026"
DEFAULT_MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
DEFAULT_PORTFOLIO = Path(str(PREFIX) + "__portfolio.json")
DEFAULT_SUMMARY = Path(str(PREFIX) + "__summary.json")
DEFAULT_REPORT = Path(str(PREFIX) + "__validation_report.json")

EXPECTED_MANIFEST_SHA256 = "dd6abda4726846f482536a235093b2ed9aa5a36b12591613c400601dcb27a84a"
EXPECTED_ROWS = 660
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_CAMPAIGN = "lib1_dedup_phase1_rerun_july2026"
EXPECTED_STAGE = "stage4_downsampling"
EXPECTED_TAG = "lib1_dedup_stage4_downsampling_july2026"
EXPECTED_MODEL_SEED = 1701
EXPECTED_SUBSET_SEEDS = (104729, 130363, 155921)
EXPECTED_PARTS = {"enhancer": 200, "promoter": 120, "intron": 120, "utr3": 120, "utr5": 100}
EXPECTED_LANES = {"primary": 400, "alternative": 180, "scratch_diagnostic": 80}

EXPECTED_CONFIGS = {
    ("enhancer", "primary", "basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036", "on", "unweighted_mse"),
    ("enhancer", "alternative", "basecfg_e53d6596a16e9f43bfe71e4ea2a364dd30237733beee9030030ecbc84f6d30a0", "on", "unweighted_mse"),
    ("enhancer", "alternative", "basecfg_3f7d963d6d647ee5eb5ee02239f1b0c992c3f33d90200d52b4e00c88e7ddd02d", "on", "unweighted_mse"),
    ("enhancer", "scratch_diagnostic", "basecfg_7bb5763f52f3678922d64e5026e75fa14b79bde606319b207a5f8b30885f87b8", "off", "unweighted_mse"),
    ("promoter", "primary", "basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d", "off", "barcode_weighted_mse"),
    ("promoter", "alternative", "basecfg_9b9293193ecdac4bffee9b00e58cfdde742789ac1c2d1d625047d4578e4fc5fe", "off", "barcode_weighted_mse"),
    ("promoter", "alternative", "basecfg_9821907e1ab3069b1657e66e9befa92e967038385a0909eb1bda10b1d2df24d0", "off", "barcode_weighted_mse"),
    ("intron", "primary", "basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86", "off", "barcode_weighted_mse"),
    ("intron", "alternative", "basecfg_5b5d2d82cef98c6e0c7522dbbc388ef4da59ee65687f40159e7c9548eb2277f3", "off", "barcode_weighted_mse"),
    ("intron", "alternative", "basecfg_6079cd38f32d3f5cf024c66fb43e7f88c2ced932f984fbebe30ba99672641b74", "off", "barcode_weighted_mse"),
    ("utr3", "primary", "basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062", "off", "barcode_weighted_mse"),
    ("utr3", "alternative", "basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05", "off", "barcode_weighted_mse"),
    ("utr3", "alternative", "basecfg_1becdea28bb6a22dbb61a48222baf1cbce413ac6e405691c9bda4b1da6253f90", "off", "barcode_weighted_mse"),
    ("utr5", "primary", "basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315", "off", "barcode_weighted_mse"),
    ("utr5", "alternative", "basecfg_e3b85c86fe400906280db9093b388bb1b74a552467120eac98e86c5202650d17", "off", "barcode_weighted_mse"),
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id_hash(values) -> str:
    return canonical_hash(sorted(str(value) for value in values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_command(command: str) -> tuple[list[str], OrderedDict[str, list[str]]]:
    tokens = shlex.split(command)
    first = next((i for i, token in enumerate(tokens) if token.startswith("--")), len(tokens))
    prefix = tokens[:first]
    options: OrderedDict[str, list[str]] = OrderedDict()
    index = first
    while index < len(tokens):
        key = tokens[index]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected positional token {key!r}")
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        if key[2:] in options:
            raise ValueError(f"Duplicate command option {key}")
        options[key[2:]] = values
    return prefix, options


def one(options: dict[str, list[str]], name: str) -> str:
    values = options.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected one --{name}; observed {values!r}")
    return values[0]


def recompute_split(row: dict, cache: dict) -> dict:
    path = Path(row["split_manifest_path"])
    resolved = str(path.resolve())
    key = ("selected", resolved, int(row["outer_oof_fold"]), int(row["train_subsample_seed"]), row["downsample_n_label"])
    if key in cache:
        return cache[key]
    hash_key = ("sha256", resolved)
    if hash_key not in cache:
        cache[hash_key] = sha256_file(path)
    if cache[hash_key] != row["split_manifest_sha256"]:
        raise ValueError(f"Split-manifest SHA mismatch for row {row['row']}")
    manifest_key = ("manifest", resolved)
    if manifest_key not in cache:
        cache[manifest_key] = json.loads(path.read_text())
    manifest = cache[manifest_key]
    outer = int(row["outer_oof_fold"])
    inner = (outer + 1) % int(manifest["n_development_folds"])
    base_key = ("base", resolved, outer)
    if base_key not in cache:
        assignments = [item for item in manifest["assignments"] if item["partition"] != "audit_test"]
        outer_ids = sorted(
            str(item["construct_id"]) for item in assignments
            if item["partition"] == "development" and int(item["development_fold"]) == outer
        )
        inner_ids = sorted(
            str(item["construct_id"]) for item in assignments
            if item["partition"] == "development" and int(item["development_fold"]) == inner
        )
        pool_ids = sorted(
            str(item["construct_id"]) for item in assignments
            if int(item["n_barcodes"]) >= 1
            and (
                item["partition"] == "train_only"
                or (
                    item["partition"] == "development"
                    and int(item["development_fold"]) not in {outer, inner}
                )
            )
        )
        cache[base_key] = (outer_ids, inner_ids, pool_ids)
    outer_ids, inner_ids, pool_ids = cache[base_key]
    if row["downsample_n_label"] == "full":
        selected_ids = pool_ids
    else:
        n = int(row["train_size_n"])
        perm = np.random.default_rng(int(row["train_subsample_seed"])).permutation(len(pool_ids))
        selected_ids = [pool_ids[int(index)] for index in perm[:n]]
    result = {
        "inner": inner,
        "pool_ids": pool_ids,
        "selected_ids": selected_ids,
        "inner_ids": inner_ids,
        "outer_ids": outer_ids,
        "final_hash": manifest["expected"]["audit_ids_sha256"],
    }
    cache[key] = result
    return result


def expected_graph(row: dict) -> str:
    if row["training_regime"] == "transfer":
        return "CNNBassetBranchedScopedTransfer"
    if row["loss_mode"] == "barcode_weighted_mse":
        return "CNNWeightedRegressionTraining"
    return "CNNBasicTraining"


def validate_command(row: dict) -> None:
    prefix, options = parse_command(row["train_command"])
    if prefix[-1:] != ["train_wandb_log.py"]:
        raise ValueError(f"Row {row['row']} has wrong entry point")
    expected = {
        "campaign_id": EXPECTED_CAMPAIGN,
        "campaign_stage": EXPECTED_STAGE,
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": str(row["outer_oof_fold"]),
        "split_fold": str(row["outer_oof_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": str(EXPECTED_MODEL_SEED),
        "loss_mode": row["loss_mode"],
        "artifact_retention": "none",
        "evaluate_test_after_fit": "false",
        "manifest_mode": "development_inner_oof",
        "train_min_barcodes": "1",
        "train_size_frac": "1.0",
        "train_sampling_mode": "random",
        "train_subsample_seed": str(row["train_subsample_seed"]),
        "use_reverse_complements": "true" if row["rc_mode"] == "on" else "false",
        "barcode_weighting": "true" if row["loss_mode"] == "barcode_weighted_mse" else "false",
        "graph_module": expected_graph(row),
        "logger_type": "wandb",
        "logger_project": row["logger_project"],
        "wandb_entity": EXPECTED_ENTITY,
        "wandb_group": row["wandb_group"],
        "wandb_job_type": "stage4_downsampling_cell",
        "run_name": row["planned_run_name"],
        "exact_run_name": "true",
        "default_root_dir": row["default_root_dir"],
        "enable_progress_bar": "false",
    }
    for name, value in expected.items():
        observed = one(options, name)
        if observed != value:
            raise ValueError(f"Row {row['row']} --{name}: {observed!r} != {value!r}")
    if options.get("prediction_splits") != ["oof"]:
        raise ValueError(f"Row {row['row']} does not export OOF only")
    if options.get("epoch_eval_splits") != ["train", "val"]:
        raise ValueError(f"Row {row['row']} does not evaluate train/inner-val only")
    if row["downsample_n_label"] == "full":
        if "train_size_n" in options or int(row["subset_replicate"]) != 0:
            raise ValueError(f"Full row {row['row']} carries finite-N state")
    elif one(options, "train_size_n") != str(row["train_size_n"]):
        raise ValueError(f"Row {row['row']} has wrong finite N")
    forbidden_options = {
        "audit_ids", "audit_id_path", "predict_test", "test_prediction_output_dir",
    }
    if forbidden_options & set(options):
        raise ValueError(f"Row {row['row']} exposes forbidden final-test options")
    if any(
        value.lower() == "test"
        for name in ("prediction_splits", "epoch_eval_splits")
        for value in options.get(name, [])
    ):
        raise ValueError(f"Row {row['row']} exposes final-test evaluation")
    if row["loss_mode"] == "barcode_weighted_mse":
        if row["training_regime"] == "scratch" and one(options, "weighted_loss_reduction") != "mean":
            raise ValueError(f"Row {row['row']} changed weighted reduction")
        if one(options, "barcode_weight_cap") != "8.0" or one(options, "barcode_weight_min") != "0.1":
            raise ValueError(f"Row {row['row']} changed frozen barcode weights")


def validate(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.manifest)
    portfolio = json.loads(args.portfolio.read_text())
    summary = json.loads(args.summary.read_text())
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest SHA changed: {manifest_sha}")
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows; found {len(rows)}")
    if [int(row["row"]) for row in rows] != list(range(1, EXPECTED_ROWS + 1)):
        raise ValueError("Manifest row numbers are not contiguous")
    for field in ("cell_id", "row_fingerprint", "planned_run_name", "default_root_dir"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {field}")
    if Counter(row["part_slug"] for row in rows) != Counter(EXPECTED_PARTS):
        raise ValueError("Per-part accounting changed")
    if Counter(row["stage4_lane"] for row in rows) != Counter(EXPECTED_LANES):
        raise ValueError("Per-lane accounting changed")

    observed_configs = {
        (row["part_slug"], row["stage4_lane"], row["base_config_id"], row["rc_mode"], row["loss_mode"])
        for row in rows
    }
    if observed_configs != EXPECTED_CONFIGS:
        raise ValueError("Stage 4 portfolio/config policies changed")
    portfolio_configs = {
        (row["part_slug"], row["stage4_lane"], row["base_config_id"], row["rc_mode"], row["loss_mode"])
        for row in portfolio["configs"]
    }
    if portfolio_configs != EXPECTED_CONFIGS or portfolio.get("config_count") != 15:
        raise ValueError("Portfolio artifact disagrees with manifest")

    split_cache = {}
    shared = defaultdict(set)
    selected_sets = {}
    source_manifests = {}
    file_hashes = {}
    for row in rows:
        if row["campaign_id"] != EXPECTED_CAMPAIGN or row["campaign_stage"] != EXPECTED_STAGE:
            raise ValueError(f"Row {row['row']} changed campaign identity")
        if row["manifest_tag"] != EXPECTED_TAG or row["model_seed"] != EXPECTED_MODEL_SEED:
            raise ValueError(f"Row {row['row']} changed frozen tag/seed")
        if row["manifest_mode"] != "development_inner_oof":
            raise ValueError(f"Row {row['row']} changed Stage 4 data mode")
        if row["wandb_entity"] != EXPECTED_ENTITY:
            raise ValueError(f"Row {row['row']} changed W&B entity")
        expected_project = f"{row['part_slug']}__bashor_in_house__dedup_exact_v1__stage4_downsampling_development"
        if row["logger_project"] != expected_project:
            raise ValueError(f"Row {row['row']} changed W&B project")
        dataset_path = str(Path(row["dataset_path"]).resolve())
        if dataset_path not in file_hashes:
            file_hashes[dataset_path] = sha256_file(Path(dataset_path))
        if file_hashes[dataset_path] != row["dataset_sha256"]:
            raise ValueError(f"Dataset SHA mismatch for row {row['row']}")
        source_path = Path(row["source_command_manifest"])
        if str(source_path) not in source_manifests:
            source_manifests[str(source_path)] = read_jsonl(source_path)
        if sum(
            candidate.get("row_fingerprint") == row["source_command_row_fingerprint"]
            for candidate in source_manifests[str(source_path)]
        ) != 1:
            raise ValueError(f"Row {row['row']} lost its exact command source")

        plan = recompute_split(row, split_cache)
        expected = {
            "inner_validation_fold": plan["inner"],
            "expected_pool_n": len(plan["pool_ids"]),
            "expected_pool_id_hash": stable_id_hash(plan["pool_ids"]),
            "expected_train_n": len(plan["selected_ids"]),
            "expected_train_id_hash": stable_id_hash(plan["selected_ids"]),
            "expected_normalization_id_hash": stable_id_hash(plan["selected_ids"]),
            "expected_inner_val_n": len(plan["inner_ids"]),
            "expected_inner_val_id_hash": stable_id_hash(plan["inner_ids"]),
            "expected_oof_n": len(plan["outer_ids"]),
            "expected_oof_id_hash": stable_id_hash(plan["outer_ids"]),
            "final_test_exclusion_id_hash": plan["final_hash"],
        }
        for field, value in expected.items():
            if row[field] != value:
                raise ValueError(f"Row {row['row']} {field} changed")
        if set(plan["selected_ids"]) & (set(plan["inner_ids"]) | set(plan["outer_ids"])):
            raise ValueError(f"Row {row['row']} has train/evaluation leakage")
        if set(plan["inner_ids"]) & set(plan["outer_ids"]):
            raise ValueError(f"Row {row['row']} has inner/outer leakage")
        if row["downsample_n_label"] != "full":
            selected_sets[(row["part_slug"], row["outer_oof_fold"], row["train_subsample_seed"], int(row["train_size_n"]))] = set(plan["selected_ids"])
        shared[(row["part_slug"], row["outer_oof_fold"], row["downsample_n_label"], row["subset_replicate"], row["train_subsample_seed"])].add(row["expected_train_id_hash"])

        identity = {
            "manifest_tag": row["manifest_tag"],
            "part_slug": row["part_slug"],
            "stage4_lane": row["stage4_lane"],
            "base_config_id": row["base_config_id"],
            "outer_oof_fold": row["outer_oof_fold"],
            "inner_validation_fold": row["inner_validation_fold"],
            "downsample_n_label": row["downsample_n_label"],
            "subset_replicate": row["subset_replicate"],
            "train_subsample_seed": row["train_subsample_seed"],
            "model_seed": row["model_seed"],
            "rc_mode": row["rc_mode"],
            "loss_mode": row["loss_mode"],
        }
        expected_cell = "stage4cell_" + canonical_hash(identity)[:20]
        if row["cell_id"] != expected_cell:
            raise ValueError(f"Row {row['row']} cell ID changed")
        fingerprint = canonical_hash({
            **identity,
            "cell_id": row["cell_id"],
            "command": row["train_command"],
            "expected_pool_id_hash": row["expected_pool_id_hash"],
            "expected_train_id_hash": row["expected_train_id_hash"],
            "expected_inner_val_id_hash": row["expected_inner_val_id_hash"],
            "expected_oof_id_hash": row["expected_oof_id_hash"],
        })
        if row["row_fingerprint"] != fingerprint:
            raise ValueError(f"Row {row['row']} fingerprint changed")
        if hashlib.sha256(row["train_command"].encode("utf-8")).hexdigest() != row["train_command_sha256"]:
            raise ValueError(f"Row {row['row']} command SHA changed")
        validate_command(row)

    if any(len(values) != 1 for values in shared.values()):
        raise ValueError("Shared subset hashes differ across configurations")
    for part in EXPECTED_PARTS:
        for fold in range(5):
            for seed in EXPECTED_SUBSET_SEEDS:
                prior = set()
                for size in (40, 250, 400, 2500, 4000):
                    current = selected_sets.get((part, fold, seed, size))
                    if current is None:
                        continue
                    if prior and not prior.issubset(current):
                        raise ValueError(f"Nested-prefix failure for {(part, fold, seed, size)}")
                    prior = current

    if summary.get("manifest_sha256") != manifest_sha or summary.get("rows") != EXPECTED_ROWS:
        raise ValueError("Summary does not bind the manifest")
    if summary.get("commands_executed") != 0:
        raise ValueError("Dry-run summary claims commands were executed")
    if summary.get("final_test_loader_instantiated") is not False or summary.get("final_test_metrics_read") is not False:
        raise ValueError("Summary does not preserve final-test isolation")
    return {
        "schema_version": "lib1_dedup_stage4_validation_v1",
        "status": "valid",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "rows": len(rows),
        "configs": len(observed_configs),
        "rows_by_part": dict(sorted(Counter(row["part_slug"] for row in rows).items())),
        "rows_by_lane": dict(sorted(Counter(row["stage4_lane"] for row in rows).items())),
        "nested_prefix_tracks_checked": len({(key[0], key[1], key[2]) for key in selected_sets}),
        "final_test_loader_instantiated": False,
        "final_test_metrics_read": False,
        "commands_executed": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Lib1 dedup Stage 4 manifest validation passed")
    print(f"  manifest SHA256: {report['manifest_sha256']}")
    print(f"  rows: {report['rows']}")
    print(f"  configs: {report['configs']}")
    print(f"  nested tracks checked: {report['nested_prefix_tracks_checked']}")
    print("  final-test loader: not instantiated")
    print("  commands executed: 0")


if __name__ == "__main__":
    main()
