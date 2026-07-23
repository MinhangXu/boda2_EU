#!/usr/bin/env python3
"""Generate the frozen Lib1 dedup Stage 4 downsampling dry-run products.

The generator is development-only. It reads frozen Stage 2/3 development
artifacts and split assignments, constructs deterministic nested training-ID
prefixes, and writes commands for the Stage-4-only inner-validation/outer-OOF
data mode. It never imports a DataModule, instantiates a loader, reads a final-
test result, or launches training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_ROOT = HERE / "outputs"
MANIFEST_ROOT = OUTPUT_ROOT / "hpo_manifests"
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis" / "lib1_dedup_stage3_weighted_loss_july2026"

SELECTION_PATH = ANALYSIS_ROOT / "stage3_selected_part_policies.json"
ADMISSIBILITY_PATH = ANALYSIS_ROOT / "stage3_arm_admissibility.csv"
STAGE3_PATH = MANIFEST_ROOT / "lib1_dedup_stage3_weighted_loss_july2026__analysis_manifest.jsonl"
STAGE2_PATH = MANIFEST_ROOT / "lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
AMENDMENT_PATH = (
    REPO
    / "plan/phase1_lib1/dedup_phase1_rerun_july2026"
    / "lib1_dedup_stage4_downsampling_protocol_amendment_july17_2026.md"
)

MANIFEST_TAG = "lib1_dedup_stage4_downsampling_july2026"
CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "stage4_downsampling"
MANIFEST_STATUS = "frozen_dry_run_not_launched"
ENTITY = "minhangxu1998-baylor-college-of-medicine"
MODEL_SEED = 1701
FOLDS = tuple(range(5))
FINITE_SUBSET_SEEDS = (104729, 130363, 155921)
PRIMARY_SIZES = (40, 250, 400, 2500, 4000, "full")
ALTERNATIVE_SIZES = (40, 400, 4000, "full")
PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}
LANE_ORDER = {"primary": 0, "alternative": 1, "scratch_diagnostic": 2}
EXPECTED_ROWS = 660

DEFAULT_PREFIX = MANIFEST_ROOT / MANIFEST_TAG
DEFAULT_MANIFEST = Path(str(DEFAULT_PREFIX) + "__dry_run_manifest.jsonl")
DEFAULT_CSV = Path(str(DEFAULT_PREFIX) + "__dry_run_manifest.csv")
DEFAULT_PORTFOLIO = Path(str(DEFAULT_PREFIX) + "__portfolio.json")
DEFAULT_SUMMARY = Path(str(DEFAULT_PREFIX) + "__summary.json")

EXPECTED_INPUT_HASHES = {
    SELECTION_PATH: "f0f818fb6b4b722726e5a98edb1e525f2c66f6ff1155772d07a0b0a71769464c",
    ADMISSIBILITY_PATH: "79f3cbec27dad91e0c7d3e1c707d5c824647303bb2b4e34dd9cb8efc6f20a845",
    STAGE3_PATH: "7b2d4115e697b8ac9507b3a8e1f5ce22aa55a6da8c2fb826d9b52992932d5995",
    STAGE2_PATH: "167a12b15654d8aa9ea63ca725aafe907e7337bec8846c3be41b8935403ea66a",
    AMENDMENT_PATH: "c331a44cba1ad49b8bcffc489ddf29a92fe22efcd0bfbfbbfdb1df5d9efd867d",
}

PRIMARY = {
    "enhancer": "basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036",
    "promoter": "basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d",
    "intron": "basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86",
    "utr3": "basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062",
    "utr5": "basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315",
}

ALTERNATIVES = {
    "enhancer": (
        "basecfg_e53d6596a16e9f43bfe71e4ea2a364dd30237733beee9030030ecbc84f6d30a0",
        "basecfg_3f7d963d6d647ee5eb5ee02239f1b0c992c3f33d90200d52b4e00c88e7ddd02d",
    ),
    "promoter": (
        "basecfg_9b9293193ecdac4bffee9b00e58cfdde742789ac1c2d1d625047d4578e4fc5fe",
        "basecfg_9821907e1ab3069b1657e66e9befa92e967038385a0909eb1bda10b1d2df24d0",
    ),
    "intron": (
        "basecfg_5b5d2d82cef98c6e0c7522dbbc388ef4da59ee65687f40159e7c9548eb2277f3",
        "basecfg_6079cd38f32d3f5cf024c66fb43e7f88c2ced932f984fbebe30ba99672641b74",
    ),
    "utr3": (
        "basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05",
        "basecfg_1becdea28bb6a22dbb61a48222baf1cbce413ac6e405691c9bda4b1da6253f90",
    ),
    "utr5": (
        "basecfg_e3b85c86fe400906280db9093b388bb1b74a552467120eac98e86c5202650d17",
    ),
}

SCRATCH_DIAGNOSTIC = {
    "enhancer": "basecfg_7bb5763f52f3678922d64e5026e75fa14b79bde606319b207a5f8b30885f87b8"
}

POLICIES = {
    "enhancer": {"rc_mode": "on", "loss_mode": "unweighted_mse"},
    "promoter": {"rc_mode": "off", "loss_mode": "barcode_weighted_mse"},
    "intron": {"rc_mode": "off", "loss_mode": "barcode_weighted_mse"},
    "utr3": {"rc_mode": "off", "loss_mode": "barcode_weighted_mse"},
    "utr5": {"rc_mode": "off", "loss_mode": "barcode_weighted_mse"},
}
SCRATCH_POLICY = {"rc_mode": "off", "loss_mode": "unweighted_mse"}


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
            handle.write(canonical_json(row) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "row", "cell_id", "part_slug", "stage4_lane", "base_config_id",
        "architecture", "training_regime", "rc_mode", "loss_mode",
        "outer_oof_fold", "inner_validation_fold", "downsample_n_label",
        "train_size_n", "subset_replicate", "train_subsample_seed",
        "expected_train_n", "expected_train_id_hash", "expected_inner_val_n",
        "expected_inner_val_id_hash", "expected_oof_n", "expected_oof_id_hash",
        "planned_run_name", "logger_project", "wandb_group", "row_fingerprint",
        "train_command",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_command(command: str) -> tuple[list[str], OrderedDict[str, list[str]]]:
    tokens = shlex.split(command)
    first = next((i for i, token in enumerate(tokens) if token.startswith("--")), len(tokens))
    prefix = tokens[:first]
    options: OrderedDict[str, list[str]] = OrderedDict()
    index = first
    while index < len(tokens):
        key = tokens[index]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected positional command token {key!r}")
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[key[2:]] = values
    return prefix, options


def build_command(source: str, mutations: dict[str, list[str] | None]) -> str:
    prefix, options = parse_command(source)
    if prefix[-1:] != ["train_wandb_log.py"]:
        raise ValueError(f"Unexpected training entry point: {prefix}")
    for key, values in mutations.items():
        if values is None:
            options.pop(key, None)
        else:
            options[key] = [str(value) for value in values]
    tokens = prefix[:]
    for key, values in options.items():
        tokens.append(f"--{key}")
        tokens.extend(values)
    return " ".join(shlex.quote(token) for token in tokens)


def portfolio_specs() -> list[dict]:
    specs = []
    for part in PART_ORDER:
        specs.append({
            "part_slug": part,
            "stage4_lane": "primary",
            "base_config_id": PRIMARY[part],
            "policy": POLICIES[part],
            "sizes": PRIMARY_SIZES,
            "diagnostic_only": False,
        })
        for base_config_id in ALTERNATIVES[part]:
            specs.append({
                "part_slug": part,
                "stage4_lane": "alternative",
                "base_config_id": base_config_id,
                "policy": POLICIES[part],
                "sizes": ALTERNATIVE_SIZES,
                "diagnostic_only": False,
            })
        if part in SCRATCH_DIAGNOSTIC:
            specs.append({
                "part_slug": part,
                "stage4_lane": "scratch_diagnostic",
                "base_config_id": SCRATCH_DIAGNOSTIC[part],
                "policy": SCRATCH_POLICY,
                "sizes": PRIMARY_SIZES,
                "diagnostic_only": True,
            })
    return sorted(
        specs,
        key=lambda row: (
            PART_ORDER[row["part_slug"]],
            LANE_ORDER[row["stage4_lane"]],
            row["base_config_id"],
        ),
    )


def admissibility_index(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    return {
        (row["part_slug"], row["base_config_id"], row["rc_mode"], row["loss_mode"]): row
        for row in rows
    }


def validate_portfolio_inputs(specs: list[dict], admissibility: dict, selections: dict) -> None:
    if len(specs) != 15:
        raise ValueError(f"Expected 15 Stage 4 configurations, found {len(specs)}")
    if len({row["base_config_id"] for row in specs}) != 15:
        raise ValueError("Stage 4 portfolio contains duplicate base configurations")
    for spec in specs:
        key = (
            spec["part_slug"], spec["base_config_id"],
            spec["policy"]["rc_mode"], spec["policy"]["loss_mode"],
        )
        arm = admissibility.get(key)
        if arm is None:
            raise ValueError(f"Missing Stage 3 arm for {key}")
        if arm.get("admissible") != "True" or arm.get("selection_eligible") != "True":
            raise ValueError(f"Stage 4 arm is not development-admissible: {key}")
        if spec["stage4_lane"] == "primary":
            selected = selections[spec["part_slug"]]
            for field in ("base_config_id", "rc_mode", "loss_mode"):
                observed = spec["base_config_id"] if field == "base_config_id" else spec["policy"][field]
                if observed != selected[field]:
                    raise ValueError(f"Primary {spec['part_slug']} disagrees on {field}")
            if arm.get("selected_part_arm") != "True":
                raise ValueError(f"Primary {spec['part_slug']} is not the selected arm")
        elif spec["stage4_lane"] == "alternative" and arm.get("within_one_se") != "True":
            raise ValueError(f"Alternative is outside the frozen one-SE set: {key}")
        elif spec["stage4_lane"] == "scratch_diagnostic" and arm.get("within_one_se") == "True":
            raise ValueError("Enhancer scratch diagnostic must remain outside the one-SE set")


def stage3_metadata_row(
    stage3_rows: list[dict], part: str, base_config_id: str, rc_mode: str,
    loss_mode: str, fold: int,
) -> dict:
    matches = [
        row for row in stage3_rows
        if row.get("part_slug") == part
        and row.get("base_config_id") == base_config_id
        and row.get("rc_mode") == rc_mode
        and row.get("loss_mode") == loss_mode
        and int(row.get("development_fold", -1)) == fold
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one Stage 3 metadata row for {(part, base_config_id, rc_mode, loss_mode, fold)}; "
            f"found {len(matches)}"
        )
    return matches[0]


def command_source_row(metadata: dict, stage2_rows: list[dict]) -> dict:
    if metadata.get("train_command"):
        return metadata
    matches = [
        row for row in stage2_rows
        if row.get("part_slug") == metadata["part_slug"]
        and row.get("base_config_id") == metadata["base_config_id"]
        and row.get("rc_mode") == metadata["rc_mode"]
        and int(row.get("development_fold", -1)) == int(metadata["development_fold"])
        and row.get("train_command")
    ]
    # One Stage-2 fold can be an immutable Stage-1 reuse row without a launch
    # command. In that case another fold from the same base config/RC arm is a
    # valid command template because every fold/path field is overwritten below.
    if not matches:
        matches = [
            row for row in stage2_rows
            if row.get("part_slug") == metadata["part_slug"]
            and row.get("base_config_id") == metadata["base_config_id"]
            and row.get("rc_mode") == metadata["rc_mode"]
            and row.get("train_command")
        ]
        matches = sorted(matches, key=lambda row: int(row["development_fold"]))[:1]
    if len(matches) != 1:
        raise ValueError(f"Expected one source command for Stage 3 row {metadata['row_fingerprint']}")
    return matches[0]


def split_plan(split_path: Path, outer_fold: int, subset_seed: int, size) -> dict:
    manifest = json.loads(split_path.read_text())
    assignments = manifest["assignments"]
    inner_fold = (outer_fold + 1) % int(manifest["n_development_folds"])
    non_final = [row for row in assignments if row["partition"] != "audit_test"]
    outer_ids = sorted(
        str(row["construct_id"]) for row in non_final
        if row["partition"] == "development" and int(row["development_fold"]) == outer_fold
    )
    inner_ids = sorted(
        str(row["construct_id"]) for row in non_final
        if row["partition"] == "development" and int(row["development_fold"]) == inner_fold
    )
    pool_rows = [
        row for row in non_final
        if row["partition"] == "train_only"
        or (
            row["partition"] == "development"
            and int(row["development_fold"]) not in {outer_fold, inner_fold}
        )
    ]
    pool_rows = [row for row in pool_rows if int(row["n_barcodes"]) >= 1]
    pool_ids = sorted(str(row["construct_id"]) for row in pool_rows)
    if len(set(pool_ids)) != len(pool_ids):
        raise ValueError(f"Duplicate Stage 4 pool IDs in {split_path}")
    if set(pool_ids) & (set(inner_ids) | set(outer_ids)) or set(inner_ids) & set(outer_ids):
        raise ValueError(f"Stage 4 fold leakage in {split_path}")
    if size == "full":
        selected_ids = list(pool_ids)
    else:
        n = int(size)
        if n > len(pool_ids):
            raise ValueError(f"Requested N={n} from pool of {len(pool_ids)} in {split_path}")
        perm = np.random.default_rng(int(subset_seed)).permutation(len(pool_ids))
        selected_ids = [pool_ids[int(index)] for index in perm[:n]]
    return {
        "inner_fold": inner_fold,
        "pool_ids": pool_ids,
        "selected_ids": selected_ids,
        "inner_ids": inner_ids,
        "outer_ids": outer_ids,
        "final_test_exclusion_hash": manifest["expected"]["audit_ids_sha256"],
    }


def size_replicates(stage4_lane: str, size) -> list[tuple[int, int]]:
    if size == "full":
        return [(0, 0)]
    if stage4_lane in {"primary", "scratch_diagnostic"}:
        return [(index + 1, seed) for index, seed in enumerate(FINITE_SUBSET_SEEDS)]
    return [(1, FINITE_SUBSET_SEEDS[0])]


def project_for(part: str) -> str:
    return f"{part}__bashor_in_house__dedup_exact_v1__stage4_downsampling_development"


def make_command(
    source: dict, spec: dict, fold: int, inner_fold: int, size,
    subset_replicate: int, subset_seed: int, cell_id: str, run_root: Path,
    run_name: str, group: str,
) -> str:
    part = spec["part_slug"]
    weighted = spec["policy"]["loss_mode"] == "barcode_weighted_mse"
    size_tag = "full" if size == "full" else str(int(size))
    mutations = {
        "artifact_path": [str(run_root / "artifacts_disabled")],
        "best_checkpoint_dir": [str(run_root / "published_checkpoint_disabled")],
        "artifact_retention": ["none"],
        "evaluate_test_after_fit": ["false"],
        "prediction_output_dir": [str(run_root / "predictions")],
        "prediction_splits": ["oof"],
        "provenance_output_dir": [str(run_root / "provenance")],
        "logger_project": [project_for(part)],
        "wandb_entity": [ENTITY],
        "wandb_group": [group],
        "wandb_job_type": ["stage4_downsampling_cell"],
        "run_name": [run_name],
        "exact_run_name": ["true"],
        "model_seed": [str(MODEL_SEED)],
        "campaign_id": [CAMPAIGN_ID],
        "campaign_stage": [CAMPAIGN_STAGE],
        "cell_id": [cell_id],
        "rc_pair_id": None,
        "loss_pair_id": None,
        "source_unweighted_cell_id": None,
        "execution_disposition": ["launch"],
        "development_fold": [str(fold)],
        "source_run_ids": None,
        "wandb_tags": [
            CAMPAIGN_ID, CAMPAIGN_STAGE, part, spec["stage4_lane"],
            source.get("architecture_slug", source.get("architecture", "unknown")),
            f"outer{fold}", f"inner{inner_fold}", f"n{size_tag}",
            f"subsetrep{subset_replicate}", f"subsetseed{subset_seed}",
            spec["policy"]["rc_mode"], spec["policy"]["loss_mode"], f"seed{MODEL_SEED}",
        ],
        "epoch_eval_splits": ["train", "val"],
        "manifest_mode": ["development_inner_oof"],
        "split_fold": [str(fold)],
        "train_size_frac": ["1.0"],
        "train_size_n": None if size == "full" else [str(int(size))],
        "train_min_barcodes": ["1"],
        "train_max_barcodes": None,
        "train_sampling_mode": ["random"],
        "train_subsample_seed": [str(subset_seed)],
        "use_reverse_complements": ["true" if spec["policy"]["rc_mode"] == "on" else "false"],
        "barcode_weighting": ["true" if weighted else "false"],
        "default_root_dir": [str(run_root)],
        "enable_progress_bar": ["false"],
    }
    return build_command(source["train_command"], mutations)


def validate_rows(rows: list[dict], selected_sets: dict[tuple, set[str]]) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} Stage 4 rows, found {len(rows)}")
    for field in ("row", "cell_id", "row_fingerprint", "planned_run_name", "default_root_dir"):
        values = [str(row[field]) for row in rows]
        if len(values) != len(set(values)):
            raise ValueError(f"Stage 4 manifest has duplicate {field}")
    if [int(row["row"]) for row in rows] != list(range(1, EXPECTED_ROWS + 1)):
        raise ValueError("Stage 4 row numbers are not contiguous")

    expected_by_lane = {"primary": 400, "alternative": 180, "scratch_diagnostic": 80}
    if Counter(row["stage4_lane"] for row in rows) != Counter(expected_by_lane):
        raise ValueError("Stage 4 lane accounting mismatch")
    expected_by_part = {"enhancer": 200, "promoter": 120, "intron": 120, "utr3": 120, "utr5": 100}
    if Counter(row["part_slug"] for row in rows) != Counter(expected_by_part):
        raise ValueError("Stage 4 part accounting mismatch")

    shared_hashes = defaultdict(set)
    for row in rows:
        if int(row["expected_train_n"]) != (
            int(row["expected_pool_n"]) if row["downsample_n_label"] == "full" else int(row["train_size_n"])
        ):
            raise ValueError(f"Train-N mismatch in row {row['row']}")
        if row["expected_train_id_hash"] != row["expected_normalization_id_hash"]:
            raise ValueError(f"Normalization hash mismatch in row {row['row']}")
        if row["expected_train_id_hash"] == row["expected_oof_id_hash"]:
            raise ValueError(f"Train/OOF hash collision in row {row['row']}")
        options = parse_command(row["train_command"])[1]
        exact = {
            "campaign_stage": [CAMPAIGN_STAGE],
            "manifest_mode": ["development_inner_oof"],
            "evaluate_test_after_fit": ["false"],
            "prediction_splits": ["oof"],
            "epoch_eval_splits": ["train", "val"],
            "split_fold": [str(row["outer_oof_fold"])],
            "development_fold": [str(row["outer_oof_fold"])],
            "train_sampling_mode": ["random"],
            "train_min_barcodes": ["1"],
            "train_subsample_seed": [str(row["train_subsample_seed"])],
            "model_seed": [str(MODEL_SEED)],
            "logger_project": [row["logger_project"]],
            "run_name": [row["planned_run_name"]],
            "cell_id": [row["cell_id"]],
        }
        for key, expected in exact.items():
            if options.get(key) != expected:
                raise ValueError(f"Row {row['row']} option {key}={options.get(key)} != {expected}")
        if row["downsample_n_label"] == "full":
            if "train_size_n" in options or int(row["subset_replicate"]) != 0:
                raise ValueError(f"Full row {row['row']} has a finite subset")
        elif options.get("train_size_n") != [str(row["train_size_n"])]:
            raise ValueError(f"Finite row {row['row']} has the wrong train_size_n")
        lowered = row["train_command"].lower()
        forbidden = ("prediction_splits test", "epoch_eval_splits test", "audit_eval", "evaluate_test_after_fit true")
        if any(token in lowered for token in forbidden):
            raise ValueError(f"Row {row['row']} contains a forbidden final-test option")
        shared_key = (
            row["part_slug"], row["outer_oof_fold"], row["downsample_n_label"],
            row["subset_replicate"], row["train_subsample_seed"],
        )
        shared_hashes[shared_key].add(row["expected_train_id_hash"])
    if any(len(values) != 1 for values in shared_hashes.values()):
        raise ValueError("Configurations do not share the same part/fold/subset IDs")

    for part in PART_ORDER:
        for fold in FOLDS:
            for seed in FINITE_SUBSET_SEEDS:
                previous = set()
                for size in (40, 250, 400, 2500, 4000):
                    key = (part, fold, seed, size)
                    if key not in selected_sets:
                        continue
                    current = selected_sets[key]
                    if previous and not previous.issubset(current):
                        raise ValueError(f"Nested-prefix failure for {key}")
                    previous = current


def generate(manifest_path: Path, csv_path: Path, portfolio_path: Path, summary_path: Path) -> dict:
    observed_inputs = {}
    for path, expected in EXPECTED_INPUT_HASHES.items():
        observed = sha256_file(path)
        observed_inputs[str(path)] = observed
        if observed != expected:
            raise ValueError(f"Frozen input hash mismatch for {path}: {observed} != {expected}")

    selections = {
        row["part_slug"]: row
        for row in json.loads(SELECTION_PATH.read_text())["part_selections"]
    }
    admissibility_rows = read_csv(ADMISSIBILITY_PATH)
    admissibility = admissibility_index(admissibility_rows)
    stage3_rows = read_jsonl(STAGE3_PATH)
    stage2_rows = read_jsonl(STAGE2_PATH)
    specs = portfolio_specs()
    validate_portfolio_inputs(specs, admissibility, selections)

    portfolio_rows = []
    rows = []
    selected_sets: dict[tuple, set[str]] = {}
    split_cache: dict[tuple[str, int, int, object], dict] = {}
    for spec in specs:
        part = spec["part_slug"]
        policy = spec["policy"]
        arm = admissibility[(part, spec["base_config_id"], policy["rc_mode"], policy["loss_mode"])]
        portfolio_rows.append({
            "portfolio_index": len(portfolio_rows) + 1,
            "part_slug": part,
            "stage4_lane": spec["stage4_lane"],
            "base_config_id": spec["base_config_id"],
            "architecture": arm["architecture"],
            "training_regime": arm["training_regime"],
            "initialization": arm["initialization"],
            "source_head": arm["source_head"],
            "unfreeze_scope": arm["unfreeze_scope"],
            "input_policy": arm["input_policy"],
            "rc_mode": policy["rc_mode"],
            "loss_mode": policy["loss_mode"],
            "stage3_portfolio_rank": int(arm["portfolio_rank"]),
            "stage3_one_se_tiebreak_rank": (
                None if not arm["one_se_tiebreak_rank"] else int(arm["one_se_tiebreak_rank"])
            ),
            "stage3_pooled_oof_pearson": float(arm["pooled_oof_pearson"]),
            "diagnostic_only": bool(spec["diagnostic_only"]),
            "sizes": list(spec["sizes"]),
            "finite_subset_seeds": (
                list(FINITE_SUBSET_SEEDS)
                if spec["stage4_lane"] in {"primary", "scratch_diagnostic"}
                else [FINITE_SUBSET_SEEDS[0]]
            ),
        })
        for fold in FOLDS:
            metadata = stage3_metadata_row(
                stage3_rows, part, spec["base_config_id"],
                policy["rc_mode"], policy["loss_mode"], fold,
            )
            source = command_source_row(metadata, stage2_rows)
            split_path = Path(source["split_manifest_path"])
            if sha256_file(split_path) != source["split_manifest_sha256"]:
                raise ValueError(f"Source split-manifest hash changed: {split_path}")
            for size in spec["sizes"]:
                for subset_replicate, subset_seed in size_replicates(spec["stage4_lane"], size):
                    cache_key = (str(split_path.resolve()), fold, subset_seed, size)
                    plan = split_cache.setdefault(
                        cache_key, split_plan(split_path, fold, subset_seed, size)
                    )
                    if size != "full":
                        selected_sets.setdefault(
                            (part, fold, subset_seed, int(size)), set(plan["selected_ids"])
                        )
                    size_label = "full" if size == "full" else str(int(size))
                    identity = {
                        "manifest_tag": MANIFEST_TAG,
                        "part_slug": part,
                        "stage4_lane": spec["stage4_lane"],
                        "base_config_id": spec["base_config_id"],
                        "outer_oof_fold": fold,
                        "inner_validation_fold": plan["inner_fold"],
                        "downsample_n_label": size_label,
                        "subset_replicate": subset_replicate,
                        "train_subsample_seed": subset_seed,
                        "model_seed": MODEL_SEED,
                        "rc_mode": policy["rc_mode"],
                        "loss_mode": policy["loss_mode"],
                    }
                    cell_id = "stage4cell_" + canonical_hash(identity)[:20]
                    short = spec["base_config_id"][8:24]
                    run_root = (
                        OUTPUT_ROOT / "hpo_runs" / MANIFEST_TAG / part
                        / spec["stage4_lane"] / spec["base_config_id"]
                        / f"outer_{fold}" / f"n_{size_label}"
                        / f"subset_{subset_replicate}_{subset_seed}"
                    )
                    run_name = (
                        f"{MANIFEST_TAG}__{part}__{spec['stage4_lane']}__{short}"
                        f"__outer{fold}__inner{plan['inner_fold']}__n{size_label}"
                        f"__sub{subset_replicate}s{subset_seed}__m{MODEL_SEED}"
                    )
                    group = (
                        f"{CAMPAIGN_ID}__stage4__{part}__{spec['stage4_lane']}__{short}"
                    )
                    command = make_command(
                        source, spec, fold, plan["inner_fold"], size,
                        subset_replicate, subset_seed, cell_id, run_root,
                        run_name, group,
                    )
                    row = {
                        **identity,
                        "row": len(rows) + 1,
                        "cell_id": cell_id,
                        "campaign_id": CAMPAIGN_ID,
                        "campaign_stage": CAMPAIGN_STAGE,
                        "manifest_status": MANIFEST_STATUS,
                        "diagnostic_only": bool(spec["diagnostic_only"]),
                        "architecture": arm["architecture"],
                        "analysis_lane": arm["analysis_lane"],
                        "challenger_family": arm["challenger_family"] if "challenger_family" in arm else metadata.get("challenger_family", ""),
                        "policy_id": arm["policy_id"],
                        "training_regime": arm["training_regime"],
                        "initialization": arm["initialization"],
                        "source_head": arm["source_head"],
                        "unfreeze_scope": arm["unfreeze_scope"],
                        "input_policy": arm["input_policy"],
                        "pretrained_artifact_sha256": metadata.get("pretrained_artifact_sha256", ""),
                        "portfolio_rank": int(arm["portfolio_rank"]),
                        "train_size_n": None if size == "full" else int(size),
                        "is_full": size == "full",
                        "expected_pool_n": len(plan["pool_ids"]),
                        "expected_pool_id_hash": stable_id_hash(plan["pool_ids"]),
                        "expected_train_n": len(plan["selected_ids"]),
                        "expected_train_id_hash": stable_id_hash(plan["selected_ids"]),
                        "expected_normalization_id_hash": stable_id_hash(plan["selected_ids"]),
                        "expected_inner_val_n": len(plan["inner_ids"]),
                        "expected_inner_val_id_hash": stable_id_hash(plan["inner_ids"]),
                        "expected_oof_n": len(plan["outer_ids"]),
                        "expected_oof_id_hash": stable_id_hash(plan["outer_ids"]),
                        "final_test_exclusion_id_hash": plan["final_test_exclusion_hash"],
                        "dataset_path": source["dataset_path"],
                        "dataset_sha256": source["dataset_sha256"],
                        "data_generation_id": source["data_generation_id"],
                        "split_manifest_path": str(split_path.resolve()),
                        "split_manifest_id": source["split_manifest_id"],
                        "split_manifest_sha256": source["split_manifest_sha256"],
                        "source_stage3_row_fingerprint": metadata["row_fingerprint"],
                        "source_command_manifest": str(
                            STAGE3_PATH.resolve() if source is metadata else STAGE2_PATH.resolve()
                        ),
                        "source_command_row_fingerprint": source["row_fingerprint"],
                        "target_definition": source["target_definition"],
                        "length_policy": source["length_policy"],
                        "artifact_retention": "none",
                        "evaluate_test_after_fit": False,
                        "prediction_splits": ["oof"],
                        "epoch_eval_splits": ["train", "val"],
                        "manifest_mode": "development_inner_oof",
                        "execution_disposition": "launch",
                        "logger_project": project_for(part),
                        "wandb_entity": ENTITY,
                        "wandb_group": group,
                        "wandb_job_type": "stage4_downsampling_cell",
                        "planned_run_name": run_name,
                        "default_root_dir": str(run_root.resolve()),
                        "train_command": command,
                        "train_command_sha256": hashlib.sha256(command.encode("utf-8")).hexdigest(),
                    }
                    row["row_fingerprint"] = canonical_hash({
                        **identity,
                        "cell_id": cell_id,
                        "command": command,
                        "expected_pool_id_hash": row["expected_pool_id_hash"],
                        "expected_train_id_hash": row["expected_train_id_hash"],
                        "expected_inner_val_id_hash": row["expected_inner_val_id_hash"],
                        "expected_oof_id_hash": row["expected_oof_id_hash"],
                    })
                    rows.append(row)

    validate_rows(rows, selected_sets)
    write_jsonl(manifest_path, rows)
    write_csv(csv_path, rows)
    portfolio = {
        "schema_version": "lib1_dedup_stage4_portfolio_v1",
        "manifest_tag": MANIFEST_TAG,
        "status": MANIFEST_STATUS,
        "primary_sizes": list(PRIMARY_SIZES),
        "alternative_sizes": list(ALTERNATIVE_SIZES),
        "finite_subset_seeds": list(FINITE_SUBSET_SEEDS),
        "model_seed": MODEL_SEED,
        "configs": portfolio_rows,
        "config_count": len(portfolio_rows),
        "primary_config_count": 5,
        "alternative_config_count": 9,
        "scratch_diagnostic_config_count": 1,
    }
    write_json(portfolio_path, portfolio)
    summary = {
        "schema_version": "lib1_dedup_stage4_manifest_summary_v1",
        "manifest_tag": MANIFEST_TAG,
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "status": MANIFEST_STATUS,
        "rows": len(rows),
        "rows_by_lane": dict(sorted(Counter(row["stage4_lane"] for row in rows).items())),
        "rows_by_part": dict(sorted(Counter(row["part_slug"] for row in rows).items())),
        "configs": len(portfolio_rows),
        "model_seed": MODEL_SEED,
        "finite_subset_seeds": list(FINITE_SUBSET_SEEDS),
        "primary_sizes": list(PRIMARY_SIZES),
        "alternative_sizes": list(ALTERNATIVE_SIZES),
        "outer_inner_rule": "outer=k; inner=(k+1)%5",
        "commands_executed": 0,
        "final_test_loader_instantiated": False,
        "final_test_metrics_read": False,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "csv_path": str(csv_path.resolve()),
        "csv_sha256": sha256_file(csv_path),
        "portfolio_path": str(portfolio_path.resolve()),
        "portfolio_sha256": sha256_file(portfolio_path),
        "frozen_input_sha256": observed_inputs,
    }
    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate(args.manifest, args.csv, args.portfolio, args.summary)
    print("Generated Lib1 dedup Stage 4 downsampling dry-run products")
    print(f"  manifest: {summary['manifest_path']}")
    print(f"  SHA256: {summary['manifest_sha256']}")
    print(f"  rows: {summary['rows']}")
    print(f"  rows by lane: {summary['rows_by_lane']}")
    print(f"  rows by part: {summary['rows_by_part']}")
    print("  commands executed: 0")


if __name__ == "__main__":
    main()
