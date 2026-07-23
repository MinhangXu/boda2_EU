import argparse
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from boda.data.bashor_datamodule import DNARegressionDataset
from boda.graph import CNNWeightedRegressionTraining
from src.learn.train_wandb_log import _validate_stage3_weighted_contract
from src.learn.generate_lib1_dedup_stage2_manifest import parse_command
from src.learn.run_lib1_dedup_stage3_campaign import (
    expected_registry_fields,
    stable_id_hash,
    validate_completed_record,
)
from src.learn.verify_lib1_dedup_stage3_manifest import validate


REPO = Path(__file__).resolve().parents[1]
LEARN = REPO / "src/learn"
PREFIX = LEARN / "outputs/hpo_manifests/lib1_dedup_stage3_weighted_loss_july2026"


def paths():
    return argparse.Namespace(
        manifest=Path(str(PREFIX) + "__dry_run_manifest.jsonl"),
        analysis_manifest=Path(str(PREFIX) + "__analysis_manifest.jsonl"),
        reuse_manifest=Path(str(PREFIX) + "__unweighted_reuse.jsonl"),
        portfolio=Path(str(PREFIX) + "__portfolio.json"),
        summary=Path(str(PREFIX) + "__summary.json"),
        stage2_analysis_manifest=(
            LEARN / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
        ),
        targeted_utr3_manifest=(
            LEARN
            / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026__dry_run_manifest.jsonl"
        ),
        stage2_metrics=(
            LEARN / "outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv"
        ),
        targeted_metrics=(
            LEARN
            / "outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/utr3_targeted_hpo_combined_arm_metrics.csv"
        ),
    )


@unittest.skipUnless(
    Path(str(PREFIX) + "__dry_run_manifest.jsonl").is_file(),
    "frozen Stage 3 manifests are unavailable",
)
class Stage3ManifestContractTests(unittest.TestCase):
    def test_exact_accounting_routes_and_audit_isolation(self):
        report = validate(paths())

        self.assertEqual(report["validation_status"], "passed")
        self.assertEqual(report["configs"], 50)
        self.assertEqual(report["new_weighted_cells"], 450)
        self.assertEqual(report["unweighted_reuse_cells"], 450)
        self.assertEqual(report["analysis_cells"], 900)
        self.assertEqual(report["complete_oof_arms"], 180)
        self.assertEqual(report["fold_level_loss_pairs"], 450)
        self.assertEqual(report["fold_level_rc_pairs"], 400)
        self.assertFalse(report["audit_loader_instantiated"])
        self.assertFalse(report["audit_ids_materialized"])
        self.assertFalse(report["audit_stratum_counts_inspected"])
        self.assertEqual(report["commands_executed"], 0)

        with paths().manifest.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(
            Counter(row["part_slug"] for row in rows),
            {
                "enhancer": 100,
                "promoter": 100,
                "intron": 100,
                "utr3": 50,
                "utr5": 100,
            },
        )
        enhancer = [row for row in rows if row["part_slug"] == "enhancer"]
        self.assertEqual(
            Counter(row["training_regime"] for row in enhancer),
            {"transfer": 60, "scratch": 40},
        )
        self.assertTrue(
            all(
                row["graph_module"] == "CNNBassetBranchedScopedWeightedTransfer"
                for row in enhancer
                if row["training_regime"] == "transfer"
            )
        )
        self.assertTrue(
            all(
                row["graph_module"] == "CNNWeightedRegressionTraining"
                for row in rows
                if row["training_regime"] == "scratch"
            )
        )

        utr3 = [row for row in rows if row["part_slug"] == "utr3"]
        self.assertEqual({row["rc_mode"] for row in utr3}, {"off"})
        self.assertTrue(all(not row["rc_pair_id"] for row in utr3))
        architectures_by_config = {
            row["base_config_id"]: row["architecture"] for row in utr3
        }
        self.assertEqual(
            Counter(architectures_by_config.values()),
            {"UTR_BassetVL": 7, "ResNet1DRegressor": 3},
        )
        self.assertTrue(all(row["evaluate_test_after_fit"] is False for row in rows))
        self.assertTrue(all(row["prediction_splits"] == ["val"] for row in rows))
        self.assertTrue(all(row["epoch_eval_splits"] == ["train", "val"] for row in rows))
        self.assertTrue(all(row["barcode_weight_cap"] == 8.0 for row in rows))
        self.assertTrue(all(row["barcode_weight_min"] == 0.1 for row in rows))

        for row in rows:
            options = parse_command(row["train_command"])
            one = lambda name, default="": options.get(name, [default])[0]
            main = argparse.Namespace(
                campaign_stage=one("campaign_stage"),
                graph_module=one("graph_module"),
                training_regime=one("training_regime"),
                loss_mode=one("loss_mode"),
                evaluate_test_after_fit=one("evaluate_test_after_fit") == "true",
                part_slug=one("part_slug"),
                logger_project=one("logger_project"),
                cell_id=one("cell_id"),
                loss_pair_id=one("loss_pair_id"),
                source_unweighted_cell_id=one("source_unweighted_cell_id"),
                rc_mode=one("rc_mode"),
                rc_pair_id=one("rc_pair_id"),
            )
            data = argparse.Namespace(
                barcode_weighting=one("barcode_weighting") == "true",
                barcode_weight_cap=float(one("barcode_weight_cap")),
                barcode_weight_min=float(one("barcode_weight_min")),
                use_reverse_complements=one("use_reverse_complements") == "true",
            )
            _validate_stage3_weighted_contract(main, data)

    def test_each_command_family_is_accepted_by_the_training_parser(self):
        with paths().manifest.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        representatives = {}
        for row in rows:
            key = (row["part_slug"], row["architecture"], row["training_regime"])
            representatives.setdefault(key, row)

        for key, row in representatives.items():
            tokens = shlex.split(row["train_command"])
            with self.subTest(command_family=key):
                result = subprocess.run(
                    [sys.executable, *tokens[1:], "--help"],
                    cwd=LEARN,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr[-2000:])


class Stage3RuntimeContractTests(unittest.TestCase):
    @staticmethod
    def weighted_graph(outputs=1):
        class IdentityModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.criterion = nn.MSELoss()

            def forward(self, value):
                return value[:, :outputs]

        return CNNWeightedRegressionTraining(
            model=IdentityModel(),
            optimizer_args={"lr": 1e-3},
            log_legacy_metric_aliases=False,
        )

    def test_strict_weighted_loss_and_invalid_weight_failures(self):
        graph = self.weighted_graph(outputs=2)
        x = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
        y = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        weights = torch.tensor([1.0, 3.0])
        expected = ((x - y).pow(2).mean(dim=1) * weights).sum() / weights.sum()
        observed = graph.training_step((x, y, weights), 0)["loss"]
        self.assertTrue(torch.allclose(observed, expected))

        with self.assertRaisesRegex(ValueError, "three-item"):
            graph.training_step((x, y), 0)
        invalid = (
            (torch.tensor([1.0]), "one weight per sample"),
            (torch.tensor([1.0, float("nan")]), "all be finite"),
            (torch.tensor([1.0, -0.1]), "must be nonnegative"),
            (torch.tensor([0.0, 0.0]), "positive finite sum"),
        )
        for invalid_weights, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    graph.training_step((x, y, invalid_weights), 0)

    def test_rc_augmentation_preserves_target_and_weight(self):
        dna = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
        targets = torch.tensor([[1.0], [2.0]])
        weights = torch.tensor([0.25, 0.75])
        dataset = DNARegressionDataset(dna, targets, weights, use_reverse_complements=True)

        original_x, original_y, original_w = dataset[0]
        rc_x, rc_y, rc_w = dataset[1]

        self.assertFalse(torch.equal(original_x, rc_x))
        self.assertTrue(torch.equal(original_y, rc_y))
        self.assertTrue(torch.equal(original_w, rc_w))

    def test_training_contract_fails_closed_on_nominal_weighting(self):
        main = argparse.Namespace(
            campaign_stage="stage3_weighted_loss",
            graph_module="CNNWeightedRegressionTraining",
            training_regime="scratch",
            loss_mode="barcode_weighted_mse",
            evaluate_test_after_fit=False,
            part_slug="promoter",
            logger_project=(
                "promoter__bashor_in_house__dedup_exact_v1__stage3_weighted_development"
            ),
            cell_id="cell_test",
            loss_pair_id="losspair_test",
            source_unweighted_cell_id="cell_source",
            rc_mode="off",
            rc_pair_id="rcpair_test",
        )
        data = argparse.Namespace(
            barcode_weighting=True,
            barcode_weight_cap=8.0,
            barcode_weight_min=0.1,
            use_reverse_complements=False,
        )
        _validate_stage3_weighted_contract(main, data)

        main.graph_module = "CNNBasicTraining"
        with self.assertRaisesRegex(ValueError, "require graph_module"):
            _validate_stage3_weighted_contract(main, data)

        main.graph_module = "CNNWeightedRegressionTraining"
        data.barcode_weighting = False
        with self.assertRaisesRegex(ValueError, "barcode_weighting=true"):
            _validate_stage3_weighted_contract(main, data)

    def test_completion_accepts_only_matching_audit_exclusion_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "predictions"
            provenance_dir = root / "provenance"
            prediction_dir.mkdir()
            provenance_dir.mkdir()
            prediction = prediction_dir / "run123__val_predictions.tsv"
            prediction.write_text(
                "construct_id\tlog2_RNA_DNA\tprediction_raw\n"
                "a\t1.0\t1.1\n"
                "b\t2.0\t1.9\n"
            )
            val_hash = stable_id_hash(["a", "b"])
            audit_hash = "precomputed_exclusion_hash"
            row = {
                "cell_id": "cell_test",
                "default_root_dir": str(root),
                "source_prediction_rows": 2,
                "source_val_row_id_hash": val_hash,
                "planned_run_name": "stage3_test_run",
                "wandb_entity": "test-entity",
                "logger_project": "enhancer__stage3_weighted_development",
                "campaign_id": "lib1_dedup_phase1_rerun_july2026",
                "campaign_stage": "stage3_weighted_loss",
                "part_slug": "enhancer",
                "analysis_lane": "test",
                "challenger_family": "none",
                "policy_id": "basecfg_test",
                "config_origin": "test",
                "training_regime": "scratch",
                "rc_pair_id": "rcpair_test",
                "loss_pair_id": "losspair_test",
                "source_unweighted_cell_id": "cell_source",
                "rc_mode": "off",
                "execution_disposition": "launch",
                "initialization": "scratch",
                "source_head": "",
                "unfreeze_scope": "",
                "input_policy": "neutral_pad216_v1",
                "pretrained_artifact_sha256": "",
                "data_generation_id": "data_v1",
                "dataset_sha256": "dataset_sha",
                "split_manifest_id": "split_v1",
                "split_manifest_sha256": "split_sha",
                "development_fold": 0,
                "base_config_id": "basecfg_test",
                "architecture": "ResNet1DRegressor",
                "model_seed": 1701,
                "loss_mode": "barcode_weighted_mse",
                "target_definition": "log2_ratio",
                "length_policy": "neutral_pad216_v1",
                "artifact_retention": "none",
            }
            run_url = (
                "https://wandb.ai/test-entity/"
                "enhancer__stage3_weighted_development/runs/run123"
            )
            provenance = {
                **expected_registry_fields(row),
                "run_id": "run123",
                "run_url": run_url,
                "status": "completed",
                "prediction_path": str(prediction),
                "data_split_summary": {
                    "n_test": 0,
                    "n_val": 2,
                    "val_row_id_hash": val_hash,
                    "audit_row_id_hash": audit_hash,
                    "data_generation_id": row["data_generation_id"],
                    "dataset_sha256": row["dataset_sha256"],
                    "split_manifest_id": row["split_manifest_id"],
                    "split_manifest_sha256": row["split_manifest_sha256"],
                    "development_fold": row["development_fold"],
                },
            }
            provenance_path = provenance_dir / "run123__run_provenance.json"
            provenance_path.write_text(json.dumps(provenance))
            record = {
                "run_id": "run123",
                "run_url": run_url,
                "prediction_path": str(prediction),
                "val_row_id_hash": val_hash,
                "audit_row_id_hash": audit_hash,
                **{field: "" for field in (
                    "test_loss", "test_r2", "test_pearson", "test_spearman",
                    "test_pearson_r2", "test_cod_r2", "test_mse",
                )},
            }

            validate_completed_record(row, record)
            record["audit_row_id_hash"] = "different"
            with self.assertRaisesRegex(RuntimeError, "audit-exclusion hash mismatch"):
                validate_completed_record(row, record)

            record["audit_row_id_hash"] = audit_hash
            provenance["loss_pair_id"] = "wrong_pair"
            provenance_path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(RuntimeError, "identity mismatches"):
                validate_completed_record(row, record)


if __name__ == "__main__":
    unittest.main()
