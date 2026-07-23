import csv
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from src.learn import train_wandb_log as training


def _append_registry_row(path: str, index: int) -> None:
    os.environ["BODA_RUNS_CSV"] = path
    training.append_runs_csv_row(
        {
            "run_id": f"run_{index}",
            "run_name": f"name_{index}",
            "status": "completed",
            "campaign_id": "lock_test",
        }
    )


class _FakeRun:
    def __init__(self, entity="expected", project="project"):
        self.entity = entity
        self.project = project
        self.id = "fake1234"
        self.summary = {}

    def get_url(self):
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.id}"

    def save(self, *args, **kwargs):
        return None


class Stage1TrainingContractTests(unittest.TestCase):
    def test_registry_append_is_process_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "runs.csv")
            processes = [
                multiprocessing.Process(target=_append_registry_row, args=(path, index))
                for index in range(8)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertEqual(process.exitcode, 0)

            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual({row["run_id"] for row in rows}, {f"run_{i}" for i in range(8)})
            self.assertTrue(all(row["campaign_id"] == "lock_test" for row in rows))

    def test_wandb_identity_mismatch_aborts(self):
        fake = _FakeRun(entity="wrong")
        with mock.patch.object(training.wandb, "run", fake), mock.patch.object(
            training.wandb, "finish"
        ) as finish:
            with self.assertRaisesRegex(RuntimeError, "entity mismatch"):
                training._assert_wandb_identity("expected", "project")
            finish.assert_called_once_with(exit_code=1)

    def test_stage1_history_contract_does_not_define_test_metrics(self):
        fake = _FakeRun()
        defined = []
        with mock.patch.object(training.wandb, "run", fake), mock.patch.object(
            training.wandb, "define_metric", side_effect=lambda key, **kwargs: defined.append(key)
        ), mock.patch.object(training.wandb, "log"):
            training._configure_wandb_history_contract(["train", "val"])
        self.assertTrue(any(key.startswith("train_") for key in defined))
        self.assertTrue(any(key.startswith("val_") for key in defined))
        self.assertFalse(any(key.startswith("test_") for key in defined))

    def test_disabled_postfit_test_never_requests_loader(self):
        class ForbiddenTestData(training.LightningDataModule):
            def test_dataloader(self):
                raise AssertionError("test loader must remain unavailable")

        trainer = mock.Mock()
        called = training._run_optional_postfit_test(
            trainer, mock.Mock(), ForbiddenTestData(), enabled=False
        )
        self.assertFalse(called)
        trainer.test.assert_not_called()

    def test_best_checkpoint_train_eval_is_summary_only(self):
        class TinyData:
            @staticmethod
            def train_eval_dataloader():
                return [(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [2.0]]))]

            train_dataloader = train_eval_dataloader

        graph = torch.nn.Linear(1, 1, bias=False)
        graph.criterion = torch.nn.MSELoss()
        graph.log_per_output_metric_details = False
        with torch.no_grad():
            graph.weight.fill_(1.0)

        fake = _FakeRun()
        with mock.patch.object(training.wandb, "run", fake), mock.patch.object(
            training.wandb, "log"
        ) as log:
            training._log_train_eval_metrics(graph, TinyData())

        log.assert_not_called()
        self.assertEqual(fake.summary["train_mse"], 0.0)
        self.assertEqual(fake.summary["best_checkpoint_train_mse"], 0.0)

    def test_prune_removes_transient_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoints" / "epoch=3-step=9.ckpt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"temporary")
            callback = type("Callback", (), {"best_model_path": str(checkpoint)})()
            removed = training.prune_lightning_checkpoints(
                {"model_checkpoint": callback}, keep=False
            )
            self.assertEqual(removed, [str(checkpoint)])
            self.assertFalse(checkpoint.exists())

    def test_prune_sweeps_explicit_checkpoint_dir_without_best_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            checkpoint_dir.mkdir()
            stale = checkpoint_dir / "stale.ckpt"
            stale.write_bytes(b"temporary")
            callback = type("Callback", (), {"best_model_path": ""})()
            removed = training.prune_lightning_checkpoints(
                {"model_checkpoint": callback},
                keep=False,
                extra_checkpoint_dirs=[str(checkpoint_dir)],
            )
            self.assertEqual(removed, [str(stale)])
            self.assertFalse(stale.exists())

    def test_dedup_campaign_entity_is_literal(self):
        namespace = training.argparse.Namespace(
            campaign_id=training.LIB1_DEDUP_CAMPAIGN_ID,
            logger_type="wandb",
            wandb_entity="wrong",
            logger_project="enhancer__bashor_in_house__dedup_exact_v1__scratch__resnet1d__exact_replay",
        )
        with self.assertRaisesRegex(ValueError, "requires --wandb_entity"):
            training._validate_campaign_wandb_contract(namespace)

    def test_targeted_utr3_campaign_project_is_explicitly_allowed(self):
        namespace = training.argparse.Namespace(
            campaign_id=training.LIB1_DEDUP_CAMPAIGN_ID,
            campaign_stage="targeted_utr3_hpo",
            logger_type="wandb",
            wandb_entity=training.LIB1_DEDUP_WANDB_ENTITY,
            logger_project=training.LIB1_DEDUP_TARGETED_UTR3_PROJECT,
        )
        training._validate_campaign_wandb_contract(namespace)

    def test_targeted_utr3_campaign_rejects_other_projects(self):
        namespace = training.argparse.Namespace(
            campaign_id=training.LIB1_DEDUP_CAMPAIGN_ID,
            campaign_stage="targeted_utr3_hpo",
            logger_type="wandb",
            wandb_entity=training.LIB1_DEDUP_WANDB_ENTITY,
            logger_project="utr3__bashor_in_house__dedup_exact_v1__stage2_development",
        )
        with self.assertRaisesRegex(ValueError, "Unexpected Lib1 dedup campaign W&B project"):
            training._validate_campaign_wandb_contract(namespace)

    def test_campaign_fields_are_flat_and_structured(self):
        namespace = training.argparse.Namespace(
            campaign_id="campaign",
            campaign_stage="stage1",
            data_generation_id="data_v1",
            dataset_sha256="a" * 64,
            split_manifest_id="split_v1",
            split_manifest_sha256="b" * 64,
            development_fold=0,
            base_config_id="base_123",
            source_run_ids=["one", "two"],
            architecture="resnet1d",
            model_seed=1701,
            loss_mode="unweighted_mse",
            target_definition="log2(RNA/DNA)",
            length_policy="modal80",
            artifact_retention="none",
            evaluate_test_after_fit=False,
        )
        fields = training._campaign_wandb_fields(namespace)
        self.assertEqual(fields["campaign_id"], "campaign")
        self.assertEqual(fields["source_run_ids"], ["one", "two"])
        self.assertFalse(fields["evaluate_test_after_fit"])


if __name__ == "__main__":
    unittest.main()
