import argparse
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from boda.common import utils
from boda.data import Lib1EnhancerDataModule
from boda.graph import (
    CNNBassetBranchedScopedTransfer,
    CNNBassetBranchedScopedWeightedTransfer,
)
from boda.model.custom_layers import GroupedLinear
from src.learn.verify_lib1_dedup_stage2_manifest import validate


REPO = Path(__file__).resolve().parents[1]
LEARN = REPO / "src/learn"
MANIFEST_PREFIX = LEARN / "outputs/hpo_manifests/lib1_dedup_stage2_july2026"
CANONICAL_ENHANCER_SPLIT = (
    LEARN / "data_manifests/splits/lib1_enhancer_dedup_exact_v1_split.json"
)
TRANSFER_ENHANCER_SPLIT = (
    LEARN
    / "data_manifests/splits/lib1_enhancer_dedup_exact_v1_transfer_mpra600_split.json"
)
ENHANCER_DATA = (
    LEARN
    / "derived_data/enhancer/bashor_in_house/"
    "lib1_enhancer_allvalid_pad216_fastqs1_5_dedup_exact__learn_ready.tsv"
)
ENHANCER_DATA_SHA = "45766cb4dcb5de9405b8fedbfb6752eb18e7c31ee4f6098aee2eb2b4ed762c45"
TRANSFER_SPLIT_SHA = "ef7712f82ae7f8cf27ff2f2984bcfff6d7007b8319d635db5afbfd53b43b4b0a"


class _TinyBranchedModel(nn.Module):
    """State-dict-compatible 3-head parent / 1-head child test double."""

    def __init__(self, heads):
        super().__init__()
        self.conv1 = nn.Linear(2, 2)
        self.conv2 = nn.Linear(2, 2)
        self.conv3 = nn.Linear(2, 2)
        self.linear1 = nn.Linear(2, 2)
        self.branched = nn.Module()
        self.branched.branched_layer_1 = GroupedLinear(2, 2, heads)
        self.output = nn.Linear(2, heads)
        self.criterion = nn.MSELoss()

    def forward(self, value):
        return self.output(value)


class Stage2TransferGraphContractTests(unittest.TestCase):
    def test_head_slice_scope_warmup_and_differential_learning_rates(self):
        parent = _TinyBranchedModel(3)
        child = _TinyBranchedModel(1)
        with torch.no_grad():
            for index, parameter in enumerate(parent.parameters(), 1):
                values = torch.arange(parameter.numel(), dtype=parameter.dtype)
                parameter.copy_(values.reshape_as(parameter) + 100 * index)

        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "parent.tar.gz"
            artifact.write_bytes(b"fixed test artifact")
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            with mock.patch.object(utils, "unpack_artifact"), mock.patch.object(
                utils, "model_fn", return_value=parent
            ):
                graph = CNNBassetBranchedScopedTransfer(
                    model=child,
                    parent_artifact=str(artifact),
                    source_head="HepG2",
                    unfreeze_scope="conv3_plus",
                    pretrained_artifact_sha256=artifact_sha,
                    head_lr=5e-4,
                    backbone_lr=1e-4,
                    transfer_weight_decay=1e-4,
                    frozen_epochs=2,
                    output_names=["log2_RNA_DNA"],
                )

        self.assertTrue(torch.equal(child.conv1.weight, parent.conv1.weight))
        self.assertTrue(
            torch.equal(
                child.branched.branched_layer_1.weight,
                parent.branched.branched_layer_1.weight[1:2],
            )
        )
        self.assertTrue(torch.equal(child.output.weight, parent.output.weight[1:2]))
        self.assertTrue(torch.equal(child.output.bias, parent.output.bias[1:2]))

        warmup_trainable = {
            name for name, parameter in child.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(warmup_trainable)
        self.assertTrue(
            all(
                name.startswith("branched.") or name.startswith("output.")
                for name in warmup_trainable
            )
        )

        optimizer = graph.configure_optimizers()
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"backbone", "head"})
        self.assertEqual(groups["backbone"]["lr"], 1e-4)
        self.assertEqual(groups["head"]["lr"], 5e-4)
        self.assertEqual(groups["backbone"]["weight_decay"], 1e-4)
        self.assertEqual(groups["head"]["weight_decay"], 1e-4)

        graph._apply_epoch_scope(epoch=2)
        final_trainable = {
            name for name, parameter in child.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(any(name.startswith("conv3.") for name in final_trainable))
        self.assertTrue(any(name.startswith("linear") for name in final_trainable))
        self.assertFalse(any(name.startswith("conv1.") for name in final_trainable))
        self.assertFalse(any(name.startswith("conv2.") for name in final_trainable))

    def test_weighted_scoped_transfer_preserves_adapter_and_uses_weights(self):
        parent = _TinyBranchedModel(3)
        child = _TinyBranchedModel(1)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "parent.tar.gz"
            artifact.write_bytes(b"fixed weighted test artifact")
            artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
            with mock.patch.object(utils, "unpack_artifact"), mock.patch.object(
                utils, "model_fn", return_value=parent
            ):
                graph = CNNBassetBranchedScopedWeightedTransfer(
                    model=child,
                    parent_artifact=str(artifact),
                    source_head="K562",
                    unfreeze_scope="full",
                    pretrained_artifact_sha256=artifact_sha,
                    frozen_epochs=2,
                    output_names=["log2_RNA_DNA"],
                    log_legacy_metric_aliases=False,
                )

        x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        y = torch.tensor([[0.0], [1.0]])
        weights = torch.tensor([1.0, 0.25])
        prediction = graph(x).detach()
        expected = (
            (prediction.sub(y).pow(2).view(-1) * weights).sum() / weights.sum()
        )
        observed = graph.training_step((x, y, weights), 0)["loss"]
        self.assertTrue(torch.allclose(observed, expected))
        self.assertEqual(graph.active_unfreeze_scope, "branched_only")
        groups = {group["name"] for group in graph.configure_optimizers().param_groups}
        self.assertEqual(groups, {"backbone", "head"})


@unittest.skipUnless(
    ENHANCER_DATA.is_file()
    and CANONICAL_ENHANCER_SPLIT.is_file()
    and TRANSFER_ENHANCER_SPLIT.is_file(),
    "canonical Lib1 Enhancer Stage 2 inputs are unavailable",
)
class Stage2TransferDataContractTests(unittest.TestCase):
    @staticmethod
    def build_data(rc_on):
        data = Lib1EnhancerDataModule(
            datafile_path=str(ENHANCER_DATA),
            sep="\t",
            sequence_column="Enhancer",
            target_column="log2_RNA_DNA",
            barcode_column="n_barcodes",
            batch_size=256,
            padded_seq_len=600,
            padding_mode="mpra_flank",
            num_workers=0,
            normalize=True,
            split_manifest_path=str(TRANSFER_ENHANCER_SPLIT),
            split_fold=0,
            split_id_column="construct_id",
            expected_data_sha256=ENHANCER_DATA_SHA,
            expected_split_sha256=TRANSFER_SPLIT_SHA,
            test_min_barcodes=8,
            train_min_barcodes=1,
            train_sampling_mode="random",
            use_reverse_complements=rc_on,
            barcode_weighting=False,
        )
        data.setup()
        return data

    def test_transfer_view_preserves_fold_and_rc_is_train_only(self):
        off = self.build_data(False)
        on = self.build_data(True)
        self.assertEqual(off.padded_seq_len, 600)
        self.assertEqual(off.padding_mode, "mpra_flank")
        self.assertEqual(set(off.df_val["construct_id"]), set(on.df_val["construct_id"]))
        self.assertEqual(set(off.df_train["construct_id"]), set(on.df_train["construct_id"]))
        self.assertEqual(len(on.dataset_train), 2 * len(off.dataset_train))
        self.assertEqual(len(on.dataset_train_eval), len(off.dataset_train_eval))
        self.assertEqual(len(on.dataset_val), len(off.dataset_val))
        self.assertIsNone(off.dataset_test)
        self.assertIsNone(on.dataset_test)
        audit_ids = set(off.df_audit["construct_id"])
        self.assertFalse(audit_ids & set(off.df_train["construct_id"]))
        self.assertFalse(audit_ids & set(off.df_val["construct_id"]))


@unittest.skipUnless(
    Path(str(MANIFEST_PREFIX) + "__run_manifest.jsonl").is_file(),
    "generated Stage 2 manifests are unavailable",
)
class Stage2ManifestContractTests(unittest.TestCase):
    def test_generated_manifest_is_exactly_660_50_610_and_paired(self):
        result = validate(
            argparse.Namespace(
                analysis_manifest=Path(
                    str(MANIFEST_PREFIX) + "__analysis_manifest.jsonl"
                ),
                run_manifest=Path(str(MANIFEST_PREFIX) + "__run_manifest.jsonl"),
                reuse_manifest=Path(
                    str(MANIFEST_PREFIX) + "__stage1_reuse_cells.jsonl"
                ),
                split_index=(
                    LEARN / "data_manifests/lib1_dedup_exact_v1_split_manifests.json"
                ),
                utr_selection=Path(
                    str(MANIFEST_PREFIX)
                    + "__utr3_utrbassetvl_selected_configs.jsonl"
                ),
            )
        )
        self.assertEqual(result["analysis_cells"], 660)
        self.assertEqual(result["stage1_reuse_cells"], 50)
        self.assertEqual(result["launch_cells"], 610)
        self.assertEqual(result["rc_pairs"], 330)
        self.assertFalse(result["audit_loader_instantiated"])


if __name__ == "__main__":
    unittest.main()
