#!/usr/bin/env python3
"""Build the fixed July-2026 Lib1 Stage-1 exact-replay queue.

This is deliberately a local, deterministic resolver.  It reads the six
approved historical sweep caches plus the completed June outer manifest; it
does not create a W&B sweep or query mutable W&B state.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover - the production env has PyYAML
    raise SystemExit(
        "PyYAML is required. Activate the boda_env conda environment before "
        "generating the replay manifest."
    ) from exc


CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "stage1_exact_replay"
DATA_GENERATION_ID = "lib1_single_part_dedup_exact_v1"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_TARGET = "log2_RNA_DNA"
TARGET_DEFINITION = "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)"
WANDB_GROUP = CAMPAIGN_ID + "__stage1_exact_replay"
MODEL_SEED = 1701
DEVELOPMENT_FOLD = 0


SOURCE_LANES = (
    {
        "lane_id": "enhancer__resnet1d",
        "part_slug": "enhancer",
        "architecture": "ResNet1DRegressor",
        "architecture_slug": "resnet1d",
        "project": "enhancer__bashor_in_house__no_flank_hq8__scratch__resnet1d_fp32",
        "sweep_id": "az1dlbv1",
        "completed": 128,
    },
    {
        "lane_id": "promoter__promoter_bassetvl",
        "part_slug": "promoter",
        "architecture": "PromoterBassetVL",
        "architecture_slug": "promoter_bassetvl",
        "project": "promoter__bashor_in_house__lib1_allvalid__scratch__promoter_bassetvl",
        "sweep_id": "vi17zxcm",
        "completed": 128,
    },
    {
        "lane_id": "intron__resnet1d",
        "part_slug": "intron",
        "architecture": "ResNet1DRegressor",
        "architecture_slug": "resnet1d",
        "project": "introns__bashor_in_house__lib1_intron_modal80__scratch__resnet1d",
        "sweep_id": "5b0njbjz",
        "completed": 126,
    },
    {
        "lane_id": "utr3__resnet1d",
        "part_slug": "utr3",
        "architecture": "ResNet1DRegressor",
        "architecture_slug": "resnet1d",
        "project": "utr3__bashor_in_house__threeprime_modal100__scratch__resnet1d_fp32",
        "sweep_id": "bnyvegba",
        "completed": 127,
    },
    {
        "lane_id": "utr5__resnet1d",
        "part_slug": "utr5",
        "architecture": "ResNet1DRegressor",
        "architecture_slug": "resnet1d",
        "project": "utr5__bashor_in_house__fiveprime_modal50__scratch__resnet1d_fp32",
        "sweep_id": "87uud4bc",
        "completed": 128,
    },
    {
        "lane_id": "utr5__utr_bassetvl",
        "part_slug": "utr5",
        "architecture": "UTR_BassetVL",
        "architecture_slug": "utr_bassetvl",
        "project": "utr5__bashor_in_house__fiveprime_modal50__scratch__utr_bassetvl_fp32",
        "sweep_id": "hs7moccj",
        "completed": 128,
    },
)

PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}

DATA_MODULES = {
    "enhancer": "Lib1EnhancerDataModule",
    "promoter": "Lib1PromoterDataModule",
    "intron": "Lib1IntronDataModule",
    "utr3": "Lib1ThreePrimeDataModule",
    "utr5": "Lib1FivePrimeDataModule",
}

# This allowlist is the formal base-config identity. Data/split, model seed,
# RC, graph/loss policy, and logging/output fields are intentionally absent.
BASE_IDENTITY_FIELDS = (
    "model_module",
    "input_len",
    "input_channels",
    "n_outputs",
    "stem_channels",
    "stem_kernel_size",
    "block_kernel_size",
    "stage_channels",
    "stage_blocks",
    "head_hidden_channels",
    "dropout_p",
    "conv1_channels",
    "conv1_kernel_size",
    "conv2_channels",
    "conv2_kernel_size",
    "conv3_channels",
    "conv3_kernel_size",
    "adaptive_pool_output_size",
    "n_linear_layers",
    "linear_channels",
    "linear_activation",
    "linear_dropout_p",
    "use_batch_norm",
    "use_weight_norm",
    "optimizer",
    "lr",
    "weight_decay",
    "amsgrad",
    "beta1",
    "beta2",
    "eps",
    "scheduler",
    "scheduler_interval",
    "scheduler_monitor",
    "T_0",
    "T_mult",
    "eta_min",
    "last_epoch",
    "batch_size",
    "max_epochs",
    "min_epochs",
    "max_steps",
    "min_steps",
    "accumulate_grad_batches",
    "gradient_clip_val",
    "gradient_clip_algorithm",
    "check_val_every_n_epoch",
    "val_check_interval",
    "limit_train_batches",
    "limit_val_batches",
    "overfit_batches",
    "fast_dev_run",
    "num_sanity_val_steps",
    "reload_dataloaders_every_n_epochs",
    "detect_anomaly",
    "benchmark",
    "enable_checkpointing",
    "multiple_trainloader_mode",
    "checkpoint_monitor",
    "stopping_mode",
    "stopping_patience",
    "precision",
    "accelerator",
    "devices",
)

MODEL_FIELDS = {
    "ResNet1DRegressor": (
        "input_len", "input_channels", "n_outputs", "stem_channels",
        "stem_kernel_size", "stage_channels", "stage_blocks",
        "block_kernel_size", "head_hidden_channels", "dropout_p",
        "use_batch_norm",
    ),
    "PromoterBassetVL": (
        "input_len", "n_outputs", "conv1_channels", "conv1_kernel_size",
        "conv2_channels", "conv2_kernel_size", "conv3_channels",
        "conv3_kernel_size", "adaptive_pool_output_size", "n_linear_layers",
        "linear_channels", "linear_activation", "linear_dropout_p",
        "use_batch_norm", "use_weight_norm",
    ),
    "UTR_BassetVL": (
        "input_len", "n_outputs", "conv1_channels", "conv1_kernel_size",
        "conv2_channels", "conv2_kernel_size", "conv3_channels",
        "conv3_kernel_size", "adaptive_pool_output_size", "n_linear_layers",
        "linear_channels", "linear_activation", "linear_dropout_p",
        "use_batch_norm", "use_weight_norm",
    ),
}

OPTIMIZER_FIELDS = ("lr", "weight_decay", "amsgrad", "beta1", "beta2", "eps")
SCHEDULER_FIELDS = (
    "scheduler_monitor", "scheduler_interval", "T_0", "T_mult", "eta_min",
    "last_epoch",
)
TRAINER_FIELDS = (
    "max_epochs", "min_epochs", "max_steps", "min_steps",
    "accumulate_grad_batches", "gradient_clip_val", "gradient_clip_algorithm",
    "check_val_every_n_epoch", "val_check_interval", "limit_train_batches",
    "limit_val_batches", "overfit_batches", "fast_dev_run",
    "num_sanity_val_steps", "reload_dataloaders_every_n_epochs",
    "detect_anomaly", "benchmark", "enable_checkpointing",
    "multiple_trainloader_mode", "accelerator", "devices", "precision",
)

EXPERIMENTAL_EXCLUDED_FIELDS = (
    "data_module", "datafile_path", "sequence_column", "target_column",
    "barcode_column", "normalize", "padding_mode", "padded_seq_len",
    "neutral_pad_char", "train_min_barcodes", "test_min_barcodes",
    "split_seed", "val_frac_within_hq", "test_frac_within_hq",
    "val_size_within_hq", "test_size_within_hq", "train_size_frac",
    "train_sampling_mode", "model_seed", "use_reverse_complements",
    "graph_module", "loss_criterion", "reduction", "barcode_weighting",
    "output_names", "logger_type", "logger_project", "run_name",
    "artifact_path", "best_checkpoint_dir", "default_root_dir",
    "epoch_eval_splits", "log_per_output_metric_details",
    "log_legacy_metric_aliases", "num_workers",
)


# W&B sweep-agent config files do not serialize every argparse default. Exact
# replay resolves and freezes those defaults explicitly instead of depending
# on whatever the model/optimizer code happens to default to in a future run.
MODEL_DEFAULTS = {
    "ResNet1DRegressor": {
        "input_len": 600,
        "input_channels": 4,
        "stem_channels": 64,
        "stem_kernel_size": 15,
        "stage_channels": [64, 128, 256],
        "stage_blocks": [2, 2, 2],
        "block_kernel_size": 7,
        "dropout_p": 0.2,
        "head_hidden_channels": 128,
        "n_outputs": 1,
        "use_batch_norm": True,
    },
    "UTR_BassetVL": {
        "input_len": 50,
        "conv1_channels": 120,
        "conv1_kernel_size": 8,
        "conv2_channels": 120,
        "conv2_kernel_size": 8,
        "conv3_channels": 120,
        "conv3_kernel_size": 8,
        "adaptive_pool_output_size": 0,
        "n_linear_layers": 1,
        "linear_channels": 40,
        "linear_activation": "ReLU",
        "linear_dropout_p": 0.2,
        "n_outputs": 1,
        "use_batch_norm": True,
        "use_weight_norm": False,
    },
    "PromoterBassetVL": {
        "input_len": 51,
        "conv1_channels": 96,
        "conv1_kernel_size": 7,
        "conv2_channels": 96,
        "conv2_kernel_size": 7,
        "conv3_channels": 64,
        "conv3_kernel_size": 5,
        "adaptive_pool_output_size": 8,
        "n_linear_layers": 1,
        "linear_channels": 96,
        "linear_activation": "ReLU",
        "linear_dropout_p": 0.3,
        "n_outputs": 1,
        "use_batch_norm": True,
        "use_weight_norm": False,
    },
}

OPTIMIZER_DEFAULTS = {
    "Adam": {
        "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
        "weight_decay": 0.0, "amsgrad": False,
    },
    "AdamW": {
        "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
        "weight_decay": 0.0, "amsgrad": False,
    },
}


def resolve_historical_snapshot(snapshot, full_run_snapshot=None):
    resolved = clean_value(dict(snapshot))
    if full_run_snapshot is not None:
        resolved = overlay_grouped_run_config(resolved, full_run_snapshot)
    model_module = resolved.get("model_module")
    if model_module not in MODEL_DEFAULTS:
        raise ValueError("Unsupported historical model_module: %r" % model_module)
    for field, default in MODEL_DEFAULTS[model_module].items():
        if resolved.get(field) is None:
            resolved[field] = clean_value(default)
    optimizer = resolved.get("optimizer", "Adam")
    if optimizer not in OPTIMIZER_DEFAULTS:
        raise ValueError("Unsupported historical optimizer: %r" % optimizer)
    resolved["optimizer"] = optimizer
    for field, default in OPTIMIZER_DEFAULTS[optimizer].items():
        if resolved.get(field) is None:
            resolved[field] = default
    for field in (
        "lr", "weight_decay", "beta1", "beta2", "eps", "eta_min",
        "dropout_p", "linear_dropout_p", "gradient_clip_val",
        "val_check_interval", "limit_train_batches", "limit_val_batches",
        "overfit_batches",
    ):
        value = resolved.get(field)
        if isinstance(value, str):
            try:
                resolved[field] = float(value)
            except ValueError:
                pass
    scheduler = normalize_scheduler(resolved.get("scheduler"))
    resolved["scheduler"] = scheduler
    if scheduler == "CosineAnnealingWarmRestarts":
        for field, default in (("T_mult", 1), ("eta_min", 0.0), ("last_epoch", -1)):
            if resolved.get(field) is None:
                resolved[field] = default
    elif scheduler is not None:
        raise ValueError("Unsupported historical scheduler: %r" % scheduler)
    return clean_value(resolved)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_value(value):
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numeric config value")
        return float(value)
    if isinstance(value, dict):
        return {str(key): clean_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [clean_value(item) for item in value]
    return value


def unwrap_wandb_yaml(path):
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    raw = yaml.load(Path(path).read_text(), Loader=loader) or {}
    resolved = {}
    for key, value in raw.items():
        if key in {"wandb_version", "_wandb"}:
            continue
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        resolved[str(key)] = clean_value(value)
    return resolved


def overlay_grouped_run_config(fallback, full_run_snapshot):
    """Flatten the trainer's fully resolved grouped W&B config export."""
    resolved = clean_value(dict(fallback))
    prefixes = (
        "Data Module args.",
        "Model Module args.",
        "Graph Module args.",
        "Optimizer args.",
        "LR Scheduler args.",
        "Criterion args.",
        "pl.Trainer.",
        "Main args.",
    )
    for key, value in full_run_snapshot.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                field = key[len(prefix):]
                if "/" not in field and field != "help":
                    resolved[field] = clean_value(value)
                break
    betas = full_run_snapshot.get("Optimizer args.betas")
    if isinstance(betas, list) and len(betas) == 2:
        resolved["beta1"] = betas[0]
        resolved["beta2"] = betas[1]
    # argparse represented devices as a string in the historical grouped
    # export even though the sweep value and runtime meaning were integer 1.
    devices = resolved.get("devices")
    if isinstance(devices, str) and devices.isdigit():
        resolved["devices"] = int(devices)
    return clean_value(resolved)


def build_local_run_config_index(learn_dir):
    """Index local authoritative run config exports in one directory scan."""
    index = {}
    wandb_dir = learn_dir / "wandb"
    with os.scandir(str(wandb_dir)) as entries:
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith("run-"):
                continue
            run_id = entry.name.rsplit("-", 1)[-1]
            path = Path(entry.path) / "files" / "config.yaml"
            if not path.is_file():
                continue
            if run_id in index:
                raise ValueError("Duplicate local W&B run config for %s" % run_id)
            index[run_id] = path
    return index


def normalize_scheduler(value):
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return str(value)


def base_identity(snapshot):
    identity = {field: snapshot.get(field) for field in BASE_IDENTITY_FIELDS}
    identity["scheduler"] = normalize_scheduler(identity.get("scheduler"))
    if identity["scheduler"] is None:
        for field in SCHEDULER_FIELDS:
            identity[field] = None
    return clean_value(identity)


def architecture_slug(model_module):
    values = {
        "ResNet1DRegressor": "resnet1d",
        "PromoterBassetVL": "promoter_bassetvl",
        "UTR_BassetVL": "utr_bassetvl",
    }
    try:
        return values[model_module]
    except KeyError:
        raise ValueError("Unsupported exact-replay model_module: %r" % model_module)


def read_runs_csv(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row.get("status") == "completed"]
    by_run_id = {row.get("run_id"): row for row in completed if row.get("run_id")}
    return completed, by_run_id


def matching_completed_run(planned_name, completed_rows):
    matches = [
        row for row in completed_rows
        if row.get("run_name") == planned_name
        or str(row.get("run_name", "")).startswith(planned_name + "_")
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one completed registry row for %s; found %d"
            % (planned_name, len(matches))
        )
    return matches[0]


def historical_val(row):
    for key in ("best_checkpoint_val_pearson", "val_pearson"):
        try:
            value = float(row.get(key, ""))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return None


def excluded_audit(snapshot):
    return {
        field: snapshot[field]
        for field in sorted(snapshot)
        if field not in BASE_IDENTITY_FIELDS
    }


def load_broad_candidates(learn_dir, completed_rows, by_run_id, run_config_index):
    candidates = []
    counts = {}
    for lane in SOURCE_LANES:
        sweep_dir = learn_dir / "wandb" / ("sweep-" + lane["sweep_id"])
        paths = sorted(sweep_dir.glob("config-*.yaml"))
        lane_candidates = []
        for path in paths:
            run_id = path.stem[len("config-"):]
            registry = by_run_id.get(run_id)
            if registry is None or registry.get("wandb_sweep_id") != lane["sweep_id"]:
                continue
            if registry.get("wandb_project") != lane["project"]:
                raise ValueError(
                    "Approved run %s resolved to unexpected project %s"
                    % (run_id, registry.get("wandb_project"))
                )
            if run_id not in run_config_index:
                raise ValueError("Missing full local W&B config export for %s" % run_id)
            full_run_path = run_config_index[run_id]
            full_run_snapshot = unwrap_wandb_yaml(full_run_path)
            snapshot = resolve_historical_snapshot(
                unwrap_wandb_yaml(path), full_run_snapshot
            )
            if snapshot.get("model_module") != lane["architecture"]:
                raise ValueError("Architecture mismatch in %s" % path)
            identity = base_identity(snapshot)
            lane_candidates.append({
                "candidate_kind": "completed_broad_sweep_run",
                "lane_id": lane["lane_id"],
                "part_slug": lane["part_slug"],
                "architecture": lane["architecture"],
                "architecture_slug": lane["architecture_slug"],
                "source_project": lane["project"],
                "source_sweep_id": lane["sweep_id"],
                "source_run_ids": [run_id],
                "source_config_paths": [
                    str(path.resolve()), str(full_run_path.resolve())
                ],
                "source_run_config_sha256": sha256_file(full_run_path),
                "source_full_run_config_snapshot": full_run_snapshot,
                "source_config_snapshot": snapshot,
                "base_identity": identity,
                "base_config_id": "basecfg_" + sha256_json(identity),
                "base_config_sha256": sha256_json(identity),
                "excluded_from_base_config_id": excluded_audit(snapshot),
                "historical_validation_values": (
                    [] if historical_val(registry) is None else [historical_val(registry)]
                ),
            })
        counts[lane["lane_id"]] = len(lane_candidates)
        if len(lane_candidates) != lane["completed"]:
            raise ValueError(
                "Completed source gate failed for %s: expected %d, resolved %d"
                % (lane["lane_id"], lane["completed"], len(lane_candidates))
            )
        print(
            "  source lane %s: %d completed configs"
            % (lane["lane_id"], len(lane_candidates)),
            flush=True,
        )
        candidates.extend(lane_candidates)
    return candidates, counts


def load_outer_candidates(learn_dir, completed_rows, run_config_index):
    manifest_dir = learn_dir / "outputs" / "hpo_manifests"
    base_path = manifest_dir / "lib1_outer_seed_prior_no_rc_june2026__base_configs.jsonl"
    run_path = manifest_dir / "lib1_outer_seed_prior_no_rc_june2026__run_manifest.jsonl"
    base_rows = [json.loads(line) for line in base_path.open() if line.strip()]
    run_rows = [json.loads(line) for line in run_path.open() if line.strip()]
    if len(base_rows) != 120 or len(run_rows) != 600:
        raise ValueError(
            "June outer source gate failed: expected 120 bases/600 rows, found %d/%d"
            % (len(base_rows), len(run_rows))
        )

    grouped = defaultdict(list)
    for row in run_rows:
        grouped[(row["part_slug"], row["config_id"])].append(row)

    candidates = []
    part_counts = Counter()
    for snapshot in base_rows:
        base_file_snapshot = resolve_historical_snapshot(snapshot)
        key = (snapshot["part_slug"], snapshot["config_id"])
        planned = grouped.get(key, [])
        if len(planned) != 5:
            raise ValueError("Outer base %r does not have five planned rows" % (key,))
        completed = [
            matching_completed_run(row["planned_run_name"], completed_rows)
            for row in planned
        ]
        run_ids = sorted(row["run_id"] for row in completed)
        missing_exports = [run_id for run_id in run_ids if run_id not in run_config_index]
        if missing_exports:
            raise ValueError(
                "Missing full local W&B config exports for outer runs %r"
                % missing_exports
            )
        full_run_paths = [run_config_index[run_id] for run_id in run_ids]
        full_run_snapshots = [unwrap_wandb_yaml(path) for path in full_run_paths]
        resolved_runs = [
            resolve_historical_snapshot(snapshot, full_run_snapshot)
            for full_run_snapshot in full_run_snapshots
        ]
        resolved_identities = [base_identity(value) for value in resolved_runs]
        if len({canonical_json(value) for value in resolved_identities}) != 1:
            raise ValueError("Outer base %r changed config across its five runs" % (key,))
        # The compact June base file intentionally omitted argparse defaults.
        # Compare every explicitly recorded identity field, then use the full
        # completed-run export as the authoritative resolved snapshot.
        file_identity = base_identity(base_file_snapshot)
        for field in BASE_IDENTITY_FIELDS:
            if field in snapshot and snapshot[field] is not None:
                if file_identity[field] != resolved_identities[0][field]:
                    raise ValueError(
                        "Outer base field %s does not match completed runs for %r"
                        % (field, key)
                    )
        snapshot = resolved_runs[0]
        values = [value for value in (historical_val(row) for row in completed) if value is not None]
        model_module = snapshot["model_module"]
        identity = base_identity(snapshot)
        parent_run_id = snapshot.get("source_run_id")
        completed_ids = {row.get("run_id") for row in completed_rows}
        if parent_run_id and parent_run_id not in completed_ids:
            raise ValueError(
                "Outer base %r refers to non-completed parent run %s"
                % (key, parent_run_id)
            )
        all_source_run_ids = sorted(
            set(run_ids + ([parent_run_id] if parent_run_id else []))
        )
        candidates.append({
            "candidate_kind": "completed_june_outer_base",
            "lane_id": snapshot["part_slug"] + "__" + architecture_slug(model_module),
            "part_slug": snapshot["part_slug"],
            "architecture": model_module,
            "architecture_slug": architecture_slug(model_module),
            "source_project": snapshot.get("logger_project"),
            "source_sweep_id": "",
            "source_run_ids": all_source_run_ids,
            "source_outer_run_ids": run_ids,
            "source_parent_run_id": parent_run_id,
            "source_config_paths": [str(base_path.resolve())] + [
                str(path.resolve()) for path in full_run_paths
            ],
            "source_run_config_sha256": [
                sha256_file(path) for path in full_run_paths
            ],
            "source_full_run_config_snapshots": full_run_snapshots,
            "source_config_snapshot": clean_value(snapshot),
            "base_identity": identity,
            "base_config_id": "basecfg_" + sha256_json(identity),
            "base_config_sha256": sha256_json(identity),
            "excluded_from_base_config_id": excluded_audit(snapshot),
            "historical_validation_values": values,
            "outer_config_id": snapshot["config_id"],
        })
        part_counts[snapshot["part_slug"]] += 1
    expected = {"promoter": 30, "intron": 30, "utr3": 30, "utr5": 30}
    if dict(part_counts) != expected:
        raise ValueError("Unexpected outer base counts: %r" % dict(part_counts))
    print(
        "  June outer bases: %s" % json.dumps(dict(sorted(part_counts.items()))),
        flush=True,
    )
    return candidates, dict(part_counts)


def merge_exact_candidates(candidates):
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate["part_slug"], candidate["base_config_id"])].append(candidate)
    merged = []
    for (part_slug, base_id), group in grouped.items():
        identities = {canonical_json(item["base_identity"]) for item in group}
        architectures = {item["architecture"] for item in group}
        if len(identities) != 1 or len(architectures) != 1:
            raise ValueError("Base ID collision for %s/%s" % (part_slug, base_id))
        representative = sorted(
            group,
            key=lambda item: (item["candidate_kind"] != "completed_june_outer_base", item["source_run_ids"]),
        )[0]
        source_ids = sorted({run_id for item in group for run_id in item["source_run_ids"]})
        values = [value for item in group for value in item["historical_validation_values"]]
        source_references = [
            {
                "candidate_kind": item["candidate_kind"],
                "lane_id": item["lane_id"],
                "source_project": item.get("source_project"),
                "source_sweep_id": item.get("source_sweep_id"),
                "source_run_ids": item["source_run_ids"],
                "source_config_paths": item["source_config_paths"],
                "source_run_config_sha256": item["source_run_config_sha256"],
                "outer_config_id": item.get("outer_config_id"),
            }
            for item in group
        ]
        merged.append({
            "part_slug": part_slug,
            "lane_id": representative["lane_id"],
            "architecture": representative["architecture"],
            "architecture_slug": representative["architecture_slug"],
            "base_config_id": base_id,
            "base_config_sha256": representative["base_config_sha256"],
            "base_identity": representative["base_identity"],
            "resolved_historical_config": representative["source_config_snapshot"],
            "excluded_from_base_config_id": representative["excluded_from_base_config_id"],
            "source_run_ids": source_ids,
            "source_candidates": source_references,
            "source_candidate_kinds": sorted({item["candidate_kind"] for item in group}),
            "historical_validation_values": values,
            "historical_validation_mean": (
                statistics.mean(values) if values else None
            ),
        })
    return sorted(
        merged,
        key=lambda row: (
            PART_ORDER[row["part_slug"]], row["architecture_slug"], row["base_config_id"]
        ),
    )


def cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def add_arg(tokens, flag, value):
    if value is None:
        return
    tokens.extend(["--" + flag, cli_value(value)])


def add_many(tokens, flag, values):
    values = list(values or [])
    if values:
        tokens.append("--" + flag)
        tokens.extend(cli_value(value) for value in values)


def shell_command(tokens):
    return " ".join(shlex.quote(str(token)) for token in tokens)


def replay_row_fingerprint(row):
    fields = (
        "run_kind", "campaign_id", "campaign_stage", "part_slug", "lane_id",
        "architecture", "base_config_id", "data_generation_id",
        "dataset_sha256", "split_manifest_id", "split_manifest_sha256",
        "development_fold", "model_seed", "wandb_entity", "logger_project",
        "planned_run_name", "train_command",
    )
    return sha256_json({field: row.get(field) for field in fields})


def resolved_data_record(data_manifest, split_index, part_slug, generation):
    part_data = data_manifest["datasets"][part_slug]
    split = split_index["parts"][part_slug]
    if generation == "dedup":
        dataset = part_data["dedup"]
        return {
            "data_generation_id": dataset["data_generation_id"],
            "dataset_path": dataset["output_path"],
            "dataset_sha256": dataset["output_sha256"],
            "split_manifest_id": split["manifest_id"],
            "split_manifest_path": split["manifest_path"],
            "split_manifest_sha256": split["manifest_sha256"],
            "project_data_slug": "dedup_exact_v1",
            "run_kind": "exact_replay",
            "campaign_stage": CAMPAIGN_STAGE,
            "wandb_group": WANDB_GROUP,
        }
    dataset = part_data["pre_dedup"]
    return {
        "data_generation_id": dataset["data_generation_id"],
        "dataset_path": dataset["output_path"],
        "dataset_sha256": dataset["output_sha256"],
        "split_manifest_id": split["pre_dedup_manifest_id"],
        "split_manifest_path": split["pre_dedup_manifest_path"],
        "split_manifest_sha256": split["pre_dedup_manifest_sha256"],
        "project_data_slug": "pre_dedup_v0",
        "run_kind": "pre_dedup_calibration",
        "campaign_stage": "stage1_pre_dedup_calibration",
        "wandb_group": CAMPAIGN_ID + "__stage1_pre_dedup_calibration",
    }


def build_run_row(base, data_manifest, split_index, learn_dir, manifest_tag, generation):
    part_slug = base["part_slug"]
    data = resolved_data_record(data_manifest, split_index, part_slug, generation)
    dataset = data_manifest["datasets"][part_slug]["dedup" if generation == "dedup" else "pre_dedup"]
    snapshot = base["resolved_historical_config"]
    arch_slug = base["architecture_slug"]
    project = "%s__bashor_in_house__%s__scratch__%s__exact_replay" % (
        part_slug, data["project_data_slug"], arch_slug
    )
    run_suffix = "dedup" if generation == "dedup" else "pre_dedup_calibration"
    run_name = "%s__%s__%s__%s__%s" % (
        manifest_tag, part_slug, arch_slug, base["base_config_sha256"][:16], run_suffix
    )
    root = (
        learn_dir / "outputs" / "hpo_runs" / manifest_tag / data["run_kind"]
        / part_slug / base["base_config_sha256"]
    )

    tokens = ["python", "train_wandb_log.py"]
    fixed_main = (
        ("data_module", DATA_MODULES[part_slug]),
        ("model_module", base["architecture"]),
        ("graph_module", "CNNBasicTraining"),
        ("artifact_path", str(root / "artifacts")),
        ("best_checkpoint_dir", str(root / "published_checkpoint_disabled")),
        ("artifact_retention", "none"),
        ("evaluate_test_after_fit", False),
        ("prediction_output_dir", str(root / "predictions")),
        ("provenance_output_dir", str(root / "provenance")),
        ("checkpoint_monitor", snapshot.get("checkpoint_monitor", "val_pearson")),
        ("stopping_mode", snapshot.get("stopping_mode", "max")),
        ("stopping_patience", snapshot.get("stopping_patience")),
        ("logger_type", "wandb"),
        ("logger_project", project),
        ("wandb_entity", EXPECTED_ENTITY),
        ("wandb_group", data["wandb_group"]),
        ("wandb_job_type", data["run_kind"]),
        ("run_name", run_name),
        ("exact_run_name", True),
        ("model_seed", MODEL_SEED),
        ("campaign_id", CAMPAIGN_ID),
        ("campaign_stage", data["campaign_stage"]),
        ("data_generation_id", data["data_generation_id"]),
        ("dataset_sha256", data["dataset_sha256"]),
        ("split_manifest_id", data["split_manifest_id"]),
        ("split_manifest_sha256", data["split_manifest_sha256"]),
        ("development_fold", DEVELOPMENT_FOLD),
        ("base_config_id", base["base_config_id"]),
        ("architecture", base["architecture"]),
        ("loss_mode", "unweighted_mse"),
        ("target_definition", TARGET_DEFINITION),
        ("length_policy", dataset["length_policy"]),
    )
    for flag, value in fixed_main:
        add_arg(tokens, flag, value)
    add_many(tokens, "prediction_splits", ["val"])
    add_many(tokens, "wandb_tags", [
        CAMPAIGN_ID, data["campaign_stage"], part_slug, arch_slug,
        "fold0", "seed1701", "rc_off", "unweighted_mse",
    ])
    add_many(tokens, "epoch_eval_splits", ["train", "val"])
    add_many(tokens, "source_run_ids", base["source_run_ids"])

    data_args = (
        ("datafile_path", data["dataset_path"]),
        ("sep", "tab"),
        ("sequence_column", dataset["sequence_column"]),
        ("target_column", EXPECTED_TARGET),
        ("barcode_column", dataset["barcode_column"]),
        ("batch_size", base["base_identity"].get("batch_size")),
        ("padded_seq_len", dataset["padded_seq_len"]),
        ("padding_mode", dataset["padding_mode"]),
        ("neutral_pad_char", "N"),
        ("num_workers", snapshot.get("num_workers", 8)),
        ("normalize", True),
        ("split_manifest_path", data["split_manifest_path"]),
        ("split_fold", DEVELOPMENT_FOLD),
        ("split_id_column", dataset["split_id_column"]),
        ("expected_data_sha256", data["dataset_sha256"]),
        ("expected_split_sha256", data["split_manifest_sha256"]),
        ("test_min_barcodes", 8),
        ("train_min_barcodes", 1),
        ("train_size_frac", 1.0),
        ("train_sampling_mode", "random"),
        ("use_reverse_complements", False),
        ("barcode_weighting", False),
    )
    for flag, value in data_args:
        add_arg(tokens, flag, value)

    for flag in MODEL_FIELDS[base["architecture"]]:
        value = base["base_identity"].get(flag)
        if isinstance(value, list):
            add_many(tokens, flag, value)
        else:
            add_arg(tokens, flag, value)
    add_arg(tokens, "loss_criterion", "MSELoss")
    add_arg(tokens, "reduction", "mean")

    add_arg(tokens, "output_names", EXPECTED_TARGET)
    add_arg(tokens, "log_per_output_metric_details", False)
    add_arg(tokens, "log_legacy_metric_aliases", False)
    add_arg(tokens, "optimizer", base["base_identity"].get("optimizer"))
    for flag in OPTIMIZER_FIELDS:
        add_arg(tokens, flag, base["base_identity"].get(flag))
    scheduler = base["base_identity"].get("scheduler")
    add_arg(tokens, "scheduler", "None" if scheduler is None else scheduler)
    add_arg(tokens, "scheduler_interval", base["base_identity"].get("scheduler_interval", "epoch"))
    if scheduler is not None:
        for flag in ("scheduler_monitor", "T_0", "T_mult", "eta_min", "last_epoch"):
            add_arg(tokens, flag, base["base_identity"].get(flag))

    for flag in TRAINER_FIELDS:
        value = base["base_identity"].get(flag)
        if flag == "precision":
            value = 32
        elif flag == "accelerator":
            value = "gpu"
        elif flag == "devices":
            value = 1
        add_arg(tokens, flag, value)
    add_arg(tokens, "default_root_dir", str(root))
    add_arg(tokens, "enable_progress_bar", False)

    row = {
        "run_kind": data["run_kind"],
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": data["campaign_stage"],
        "manifest_tag": manifest_tag,
        "part": dataset["part"],
        "part_slug": part_slug,
        "lane_id": base["lane_id"],
        "architecture": base["architecture"],
        "architecture_slug": arch_slug,
        "base_config_id": base["base_config_id"],
        "base_config_sha256": base["base_config_sha256"],
        "base_identity": base["base_identity"],
        "resolved_historical_config": snapshot,
        "excluded_from_base_config_id": base["excluded_from_base_config_id"],
        "source_run_ids": base["source_run_ids"],
        "source_candidates": base["source_candidates"],
        "historical_validation_mean": base["historical_validation_mean"],
        "data_generation_id": data["data_generation_id"],
        "dataset_path": data["dataset_path"],
        "dataset_sha256": data["dataset_sha256"],
        "split_manifest_id": data["split_manifest_id"],
        "split_manifest_path": data["split_manifest_path"],
        "split_manifest_sha256": data["split_manifest_sha256"],
        "development_fold": DEVELOPMENT_FOLD,
        "model_seed": MODEL_SEED,
        "use_reverse_complements": False,
        "loss_mode": "unweighted_mse",
        "barcode_weighting": False,
        "target_column": EXPECTED_TARGET,
        "target_definition": TARGET_DEFINITION,
        "length_policy": dataset["length_policy"],
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "wandb_entity": EXPECTED_ENTITY,
        "logger_project": project,
        "wandb_group": data["wandb_group"],
        "task_family": part_slug,
        "target_family": data["data_generation_id"] + "__" + part_slug,
        "comparison_group": project,
        "planned_run_name": run_name,
        "default_root_dir": str(root),
        "train_command": shell_command(tokens),
    }
    row["row_fingerprint"] = replay_row_fingerprint(row)
    return row


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_csv(path, rows):
    fields = (
        "manifest_row", "run_kind", "part_slug", "lane_id", "architecture",
        "base_config_id", "planned_run_name", "data_generation_id",
        "dataset_sha256", "split_manifest_id", "split_manifest_sha256",
        "development_fold", "model_seed", "logger_project", "wandb_entity",
        "historical_validation_mean", "train_command",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tag", default="lib1_dedup_phase1_exact_replay_july2026")
    parser.add_argument(
        "--data-manifest", type=Path,
        default=here / "data_manifests" / "lib1_single_part_dedup_exact_v1.json",
    )
    parser.add_argument(
        "--split-index", type=Path,
        default=here / "data_manifests" / "lib1_dedup_exact_v1_split_manifests.json",
    )
    parser.add_argument("--runs-csv", type=Path, default=here / "run_registry" / "runs.csv")
    parser.add_argument(
        "--outer-base-manifest", type=Path,
        default=here / "outputs" / "hpo_manifests" / "lib1_outer_seed_prior_no_rc_june2026__base_configs.jsonl",
        help="Reserved compatibility argument; the canonical sibling manifests are used together.",
    )
    parser.add_argument("--outdir", type=Path, default=here / "outputs" / "hpo_manifests")
    return parser.parse_args()


def main():
    args = parse_args()
    learn_dir = Path(__file__).resolve().parent
    data_manifest = json.loads(args.data_manifest.read_text())
    split_index = json.loads(args.split_index.read_text())
    if data_manifest.get("data_generation_id") != DATA_GENERATION_ID:
        raise ValueError("Unexpected data-generation manifest")
    if split_index.get("manifest_id") != "lib1_dedup_exact_v1_split_manifests":
        raise ValueError("Unexpected split index")
    if split_index.get("data_manifest_sha256") != sha256_file(args.data_manifest):
        raise ValueError("Split index is not bound to the supplied data manifest bytes")

    completed_rows, by_run_id = read_runs_csv(args.runs_csv)
    run_config_index = build_local_run_config_index(learn_dir)
    broad, broad_counts = load_broad_candidates(
        learn_dir, completed_rows, by_run_id, run_config_index
    )
    outer, outer_counts = load_outer_candidates(
        learn_dir, completed_rows, run_config_index
    )
    candidates = broad + outer
    exact_bases = merge_exact_candidates(candidates)
    if len(broad) != 765 or len(outer) != 120:
        raise ValueError("Historical source totals changed: broad=%d outer=%d" % (len(broad), len(outer)))
    if not exact_bases:
        raise ValueError("No eligible trainer-inclusive exact bases resolved")

    exact_rows = [
        build_run_row(base, data_manifest, split_index, learn_dir, args.manifest_tag, "dedup")
        for base in exact_bases
    ]

    by_part = defaultdict(list)
    for base in exact_bases:
        by_part[base["part_slug"]].append(base)
    calibration_bases = []
    for part_slug in sorted(by_part, key=lambda value: PART_ORDER[value]):
        ranked = sorted(
            by_part[part_slug],
            key=lambda row: (
                row["historical_validation_mean"] is None,
                -(row["historical_validation_mean"] or -999.0),
                row["base_config_id"],
            ),
        )
        chosen = ranked[:5]
        if len(chosen) != 5 or any(row["historical_validation_mean"] is None for row in chosen):
            raise ValueError("Could not predeclare five validation-ranked calibration bases for %s" % part_slug)
        calibration_bases.extend(chosen)
    calibration_rows = [
        build_run_row(base, data_manifest, split_index, learn_dir, args.manifest_tag, "pre_dedup")
        for base in calibration_bases
    ]
    if len(calibration_rows) != 25:
        raise ValueError("Expected 25 calibration mates")

    rows = exact_rows + calibration_rows
    for index, row in enumerate(rows, 1):
        row["manifest_row"] = index
    names = [row["planned_run_name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate planned run names")

    prefix = args.outdir / args.manifest_tag
    run_jsonl = Path(str(prefix) + "__run_manifest.jsonl")
    write_jsonl(run_jsonl, rows)
    write_json(Path(str(prefix) + "__run_manifest.json"), rows)
    write_csv(Path(str(prefix) + "__run_manifest.csv"), rows)
    write_jsonl(Path(str(prefix) + "__base_configs.jsonl"), exact_bases)
    write_jsonl(Path(str(prefix) + "__source_candidates.jsonl"), candidates)
    write_jsonl(Path(str(prefix) + "__calibration_selection.jsonl"), calibration_bases)

    exact_by_part = Counter(row["part_slug"] for row in exact_rows)
    exact_by_lane = Counter(row["lane_id"] for row in exact_rows)
    summary = {
        "schema_version": "lib1_dedup_exact_replay_manifest_v1",
        "manifest_tag": args.manifest_tag,
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "wandb_entity": EXPECTED_ENTITY,
        "data_manifest_path": str(args.data_manifest.resolve()),
        "data_manifest_sha256": sha256_file(args.data_manifest),
        "split_index_path": str(args.split_index.resolve()),
        "split_index_sha256": sha256_file(args.split_index),
        "broad_completed_source_count": len(broad),
        "broad_completed_counts_by_lane": broad_counts,
        "outer_completed_base_count": len(outer),
        "outer_completed_counts_by_part": outer_counts,
        "candidate_count_before_base_dedup": len(candidates),
        "n_exact": len(exact_rows),
        "exact_counts_by_part": dict(sorted(exact_by_part.items())),
        "exact_counts_by_lane": dict(sorted(exact_by_lane.items())),
        "n_pre_dedup_calibration": len(calibration_rows),
        "n_manifest_rows": len(rows),
        "base_config_identity_fields": list(BASE_IDENTITY_FIELDS),
        "base_config_excluded_experimental_fields": list(EXPERIMENTAL_EXCLUDED_FIELDS),
        "inactive_scheduler_fields_are_null": True,
        "fixed_policy": {
            "development_fold": DEVELOPMENT_FOLD,
            "model_seed": MODEL_SEED,
            "use_reverse_complements": False,
            "loss_mode": "unweighted_mse",
            "precision": 32,
            "audit_test_available": False,
            "artifact_retention": "none",
            "epoch_eval_splits": ["train", "val"],
        },
        "calibration_selection": "top five per part by completed historical validation mean; never enters exact-model selection",
        "run_manifest_path": str(run_jsonl.resolve()),
        "run_manifest_sha256": sha256_file(run_jsonl),
    }
    summary_path = Path(str(prefix) + "__summary.json")
    write_json(summary_path, summary)

    print("Resolved fixed Lib1 dedup exact replay manifest")
    print("  approved completed broad configs: %d" % len(broad))
    print("  completed June outer base configs: %d" % len(outer))
    print("  N_exact after trainer-inclusive base-config dedup: %d" % len(exact_rows))
    print("  optional pre-dedup calibration mates: %d" % len(calibration_rows))
    print("  exact by part: %s" % json.dumps(dict(sorted(exact_by_part.items())), sort_keys=True))
    print("  manifest: %s" % run_jsonl)
    print("  manifest SHA256: %s" % summary["run_manifest_sha256"])


if __name__ == "__main__":
    main()
