#!/usr/bin/env python3
"""Verify Stage-1 pilot runs locally and in W&B before the full queue."""

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import wandb


EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_CAMPAIGN = "lib1_dedup_phase1_rerun_july2026"
CANONICAL_METRICS = (
    "loss", "mse", "pearson", "pearson_r2", "spearman", "cod_r2"
)


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=here / "outputs" / "hpo_manifests"
        / "lib1_dedup_phase1_exact_replay_july2026__run_manifest.jsonl",
    )
    parser.add_argument(
        "--runs-csv", type=Path, default=here / "run_registry" / "runs.csv"
    )
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=here / "outputs" / "hpo_runs" / "status"
        / "lib1_dedup_phase1_exact_replay_july2026",
    )
    parser.add_argument(
        "--manifest-rows", type=int, nargs="+", default=[1, 2],
        help="Completed exact-replay row numbers to verify (default: 1 2).",
    )
    parser.add_argument("--max-history-rows", type=int, default=10000)
    return parser.parse_args()


def load_manifest(path):
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {int(row["manifest_row"]): row for row in rows}


def registry_by_name(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        if row.get("status") != "completed" or not row.get("run_name"):
            continue
        grouped.setdefault(row["run_name"], []).append(row)
    return grouped


def summary_dict(run):
    try:
        return dict(run.summary._json_dict)
    except AttributeError:
        return dict(run.summary)


def validate_history(run, max_rows):
    seen = set()
    test_keys = set()
    row_count = 0
    for history in run.scan_history(page_size=1000):
        row_count += 1
        if row_count > max_rows:
            raise ValueError(
                "History scan exceeded --max-history-rows=%d" % max_rows
            )
        for key, value in history.items():
            if value is not None:
                seen.add(key)
                if key.startswith("test_") or key.startswith("epoch_end_test_"):
                    test_keys.add(key)
    required = {
        "%s_%s" % (split, metric)
        for split in ("train", "val")
        for metric in CANONICAL_METRICS
    }
    missing = sorted(required - seen)
    if missing:
        raise ValueError("Missing canonical W&B history keys: %s" % missing)
    if test_keys:
        raise ValueError("Forbidden test history keys have values: %s" % sorted(test_keys))
    lr_keys = sorted(
        key for key in seen
        if key == "learning_rate" or key.startswith("lr-") or key.startswith("lr_")
    )
    if not lr_keys:
        raise ValueError("No learning-rate history key was logged")
    return row_count, lr_keys


def validate_local_outputs(row, run_id):
    root = Path(row["default_root_dir"])
    prediction = root / "predictions" / (run_id + "__val_predictions.tsv")
    provenance = root / "provenance" / (run_id + "__run_provenance.json")
    if not prediction.is_file() or not provenance.is_file():
        raise ValueError(
            "Expected validation-prediction and provenance files for %s under %s"
            % (run_id, root)
        )
    forbidden = []
    for pattern in ("*.ckpt", "*.tar.gz", "*.pt"):
        forbidden.extend(root.rglob(pattern))
    if forbidden:
        raise ValueError("Retention=none left forbidden local model files: %s" % forbidden)
    return prediction, provenance


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_remote_files(run, run_id, summary):
    expected = (
        (run_id + "__val_predictions.tsv", "val_predictions_sha256"),
        (run_id + "__run_provenance.json", "compact_provenance_sha256"),
    )
    available = {file.name: file for file in run.files()}
    with tempfile.TemporaryDirectory() as tmp:
        for name, summary_key in expected:
            if name not in available:
                raise ValueError("W&B run is missing saved file %s" % name)
            downloaded = available[name].download(root=tmp, replace=True)
            if isinstance(downloaded, (str, Path)):
                downloaded_path = Path(downloaded)
            else:
                downloaded_path = Path(getattr(downloaded, "name", Path(tmp) / name))
            if not downloaded_path.is_absolute() and not downloaded_path.is_file():
                downloaded_path = Path(tmp) / downloaded_path
            if sha256_file(downloaded_path) != summary.get(summary_key):
                raise ValueError("Remote W&B file hash mismatch for %s" % name)


def main():
    args = parse_args()
    manifest = load_manifest(args.manifest)
    registry = registry_by_name(args.runs_csv)
    api = wandb.Api(timeout=30)
    if not getattr(api, "api_key", None):
        raise SystemExit("No W&B API key resolved; run `wandb login` first.")

    reports = []
    for number in args.manifest_rows:
        if number not in manifest:
            raise ValueError("Unknown manifest row %d" % number)
        row = manifest[number]
        if row["run_kind"] != "exact_replay":
            raise ValueError("Row %d is not an exact-replay row" % number)
        marker = args.status_dir / "done" / ("row_%d.done" % number)
        if not marker.is_file():
            raise ValueError("Missing completion marker %s" % marker)
        marker_fields = {}
        for line in marker.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                marker_fields[key] = value
        if marker_fields.get("row_fingerprint") != row["row_fingerprint"]:
            raise ValueError("Row %d completion fingerprint mismatch" % number)

        matches = registry.get(row["planned_run_name"], [])
        if not matches:
            matches = [
                candidate
                for name, candidates in registry.items()
                if name.startswith(row["planned_run_name"] + "_")
                for candidate in candidates
            ]
        if not matches:
            raise ValueError("No completed registry row for %s" % row["planned_run_name"])
        record = matches[-1]
        run_id = record.get("run_id")
        run_path = "%s/%s/%s" % (EXPECTED_ENTITY, row["logger_project"], run_id)
        run = api.run(run_path)
        if run.entity != EXPECTED_ENTITY or run.project != row["logger_project"]:
            raise ValueError("W&B identity mismatch for %s" % run_path)
        if run.state != "finished":
            raise ValueError("W&B run %s state is %r, not finished" % (run_path, run.state))
        if run.group != row["wandb_group"] or run.job_type != "exact_replay":
            raise ValueError("W&B group/job_type mismatch for %s" % run_path)
        required_tags = {
            EXPECTED_CAMPAIGN,
            "stage1_exact_replay",
            row["part_slug"],
            row["architecture_slug"],
            "fold0",
            "seed1701",
            "rc_off",
            "unweighted_mse",
        }
        if not required_tags.issubset(set(run.tags or [])):
            raise ValueError("W&B tags are incomplete for %s" % run_path)

        config = dict(run.config)
        for key, expected in (
            ("campaign_id", EXPECTED_CAMPAIGN),
            ("campaign_stage", "stage1_exact_replay"),
            ("base_config_id", row["base_config_id"]),
            ("dataset_sha256", row["dataset_sha256"]),
            ("split_manifest_sha256", row["split_manifest_sha256"]),
            ("development_fold", 0),
            ("model_seed", 1701),
            ("artifact_retention", "none"),
            ("evaluate_test_after_fit", False),
        ):
            if config.get(key) != expected:
                raise ValueError(
                    "W&B config %s mismatch for %s: %r != %r"
                    % (key, run_path, config.get(key), expected)
                )

        history_rows, lr_keys = validate_history(run, args.max_history_rows)
        summary = summary_dict(run)
        forbidden_summary = sorted(
            key for key in summary
            if key.startswith("test_") or key.startswith("epoch_end_test_")
        )
        if forbidden_summary:
            raise ValueError("Forbidden test summary keys: %s" % forbidden_summary)
        for key in (
            "best_checkpoint_val_pearson",
            "best_checkpoint_val_spearman",
            "best_checkpoint_val_cod_r2",
            "best_checkpoint_val_mse",
            "val_predictions_sha256",
            "compact_provenance_sha256",
            "fit_wall_time_seconds",
            "model_parameter_count",
            "resolved_wandb_run_url",
        ):
            if summary.get(key) is None:
                raise ValueError("Missing W&B summary field %s" % key)
        if summary.get("wandb_model_logging_enabled") is not False:
            raise ValueError("W&B model logging was not disabled")
        if summary.get("model_artifact_retained") is not False:
            raise ValueError("Run does not confirm model_artifact_retained=false")
        model_artifacts = [
            artifact for artifact in run.logged_artifacts()
            if getattr(artifact, "type", None) == "model"
        ]
        if model_artifacts:
            raise ValueError("Run logged forbidden W&B model artifacts")

        prediction, provenance = validate_local_outputs(row, run_id)
        if sha256_file(prediction) != summary["val_predictions_sha256"]:
            raise ValueError("Local validation-prediction hash mismatch")
        if sha256_file(provenance) != summary["compact_provenance_sha256"]:
            raise ValueError("Local compact-provenance hash mismatch")
        validate_remote_files(run, run_id, summary)
        reports.append({
            "manifest_row": number,
            "run_path": run_path,
            "history_rows": history_rows,
            "learning_rate_keys": lr_keys,
            "prediction": str(prediction),
            "provenance": str(provenance),
        })

    print(json.dumps({"verified": len(reports), "runs": reports}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
