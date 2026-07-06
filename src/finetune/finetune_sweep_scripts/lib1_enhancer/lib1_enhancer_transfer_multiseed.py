#!/usr/bin/env python3
"""Run cached multi-seed B1/B2/B3 transfer experiments on lib1 enhancer data."""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import sys
import tarfile
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def locate_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "boda").is_dir():
            return d
    for candidate in (
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path.cwd().resolve().parent.parent,
    ):
        if (candidate / "boda").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root containing `boda`.")


REPO_ROOT = locate_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boda.common import utils  # noqa: E402
from boda.model.basset import BassetBranched  # noqa: E402


DEFAULT_DATA_PATH = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/"
    "20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt"
)
DEFAULT_MODEL_PATH = REPO_ROOT / "tutorials" / "malinois_artifacts__20211113_021200__287348.tar.gz"
DEFAULT_CACHE_ROOT = REPO_ROOT / "src" / "finetune" / "cache" / "lib1_enhancer"

SEQUENCE_COLUMN = "Enhancers"
BARCODE_COLUMN = "n_barcodes"
TARGET_COLUMN = "RNA_DNA_Ratio_log10_scaled"
PRETRAINED_HEADS = ["K562", "HepG2", "SKNSH"]
INPUT_LEN = 600

TRAIN_MIN_BARCODES = 3
HIGH_QUALITY_MIN_BARCODES = 5
VAL_FRAC_WITHIN_HIGH_QUALITY = 0.20
TEST_FRAC_WITHIN_HIGH_QUALITY = 0.20

PRED_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 30
FREEZE_BACKBONE_EPOCHS = 2
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEFAULT_SEEDS = list(range(7, 15))


@dataclass(frozen=True)
class TargetScaler:
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def from_series(cls, series: pd.Series) -> "TargetScaler":
        mean = float(series.mean())
        std = float(series.std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: Any) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


@dataclass(frozen=True)
class FineTuneSetting:
    name: str
    use_rc_augmentation: bool = False
    use_barcode_weighting: bool = False
    b_cap: float | None = None
    min_weight: float = 0.1

    @property
    def cache_name(self) -> str:
        return self.name.replace(".", "_")


class PaddedSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "target_standardized",
        sequence_column: str = "padded_seq",
        weight_column: str | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.target_column = target_column
        self.sequence_column = sequence_column
        self.weight_column = weight_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq_tensor = utils.row_dna2tensor(row, in_column_name=self.sequence_column)
        y = torch.tensor(row[self.target_column], dtype=torch.float32).view(-1)

        if self.weight_column is None:
            return seq_tensor, y

        w = torch.tensor(float(row[self.weight_column]), dtype=torch.float32)
        return seq_tensor, y, w


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_barcode_weight(barcode_count: float, b_cap: float = 10.0, min_weight: float = 0.1) -> float:
    raw = np.log1p(float(barcode_count)) / np.log1p(float(b_cap))
    weight = min(1.0, raw)
    weight = max(min_weight, weight)
    return float(weight)


def add_barcode_weights(
    df: pd.DataFrame,
    barcode_column: str = BARCODE_COLUMN,
    b_cap: float = 10.0,
    min_weight: float = 0.1,
) -> pd.DataFrame:
    out = df.copy()
    out["sample_weight"] = out[barcode_column].map(
        lambda x: compute_barcode_weight(x, b_cap=b_cap, min_weight=min_weight)
    )
    return out


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    weight = weight.reshape(-1)
    se = (pred - target) ** 2
    return (weight * se).sum() / weight.sum().clamp_min(1e-8)


def reverse_complement_seq(seq: str) -> str:
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]


def augment_train_df_with_rc(
    train_df_padded: pd.DataFrame,
    padded_column: str = "padded_seq",
    random_seed: int = 7,
) -> pd.DataFrame:
    rc_df = train_df_padded.copy()
    rc_df[padded_column] = rc_df[padded_column].map(reverse_complement_seq)
    rc_df["is_rc_aug"] = True

    out_df = train_df_padded.copy()
    out_df["is_rc_aug"] = False

    aug_df = pd.concat([out_df, rc_df], axis=0, ignore_index=True)
    aug_df = aug_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return aug_df


def add_padded_sequences(
    df: pd.DataFrame,
    sequence_column: str = SEQUENCE_COLUMN,
    padded_seq_len: int = INPUT_LEN,
) -> pd.DataFrame:
    padded_df = df.copy()
    padded_df["padded_seq"] = padded_df.apply(
        lambda row: utils.row_pad_sequence(
            row,
            in_column_name=sequence_column,
            padded_seq_len=padded_seq_len,
        ),
        axis=1,
    )
    return padded_df


def make_loader(
    df: pd.DataFrame,
    target_column: str = "target_standardized",
    batch_size: int = TRAIN_BATCH_SIZE,
    shuffle: bool = False,
    weight_column: str | None = None,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    ds = PaddedSequenceDataset(
        df=df,
        target_column=target_column,
        sequence_column="padded_seq",
        weight_column=weight_column,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def load_checkpoint_from_tar(artifact_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(artifact_path, "r:gz") as tar_handle:
            tar_handle.extractall(tmpdir)
        checkpoint_path = Path(tmpdir) / "artifacts" / "torch_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def build_multitask_model(checkpoint: dict[str, Any], device: str) -> BassetBranched:
    model_hparams = vars(checkpoint["model_hparams"]).copy()
    model = BassetBranched(**model_hparams)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)


def slice_state_dict_for_head(state_dict: dict[str, Any], head_idx: int, n_heads: int) -> dict[str, Any]:
    single_head_state = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            single_head_state[key] = value
            continue
        if (
            (key.startswith("branched.branched_layer_") or key.startswith("output."))
            and value.ndim >= 1
            and value.shape[0] == n_heads
        ):
            single_head_state[key] = value[head_idx : head_idx + 1].clone()
        else:
            single_head_state[key] = value.clone()
    return single_head_state


def build_single_head_model(
    checkpoint: dict[str, Any],
    head_idx: int,
    device: str,
    loss_criterion: str = "MSELoss",
) -> BassetBranched:
    model_hparams = vars(checkpoint["model_hparams"]).copy()
    n_heads = int(model_hparams["n_outputs"])
    model_hparams["n_outputs"] = 1
    model_hparams["loss_criterion"] = loss_criterion
    model_hparams["loss_args"] = {"reduction": "mean"}

    model = BassetBranched(**model_hparams)
    single_head_state = slice_state_dict_for_head(
        checkpoint["model_state_dict"],
        head_idx=head_idx,
        n_heads=n_heads,
    )
    model.load_state_dict(single_head_state, strict=True)
    return model.to(device)


@torch.no_grad()
def predict_model(
    model: BassetBranched,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[:2]
            targets.append(y.cpu().numpy())
        else:
            x = batch
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
    pred_array = np.concatenate(preds, axis=0)
    target_array = np.concatenate(targets, axis=0) if targets else None
    return pred_array, target_array


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    metrics = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        metrics["pearson"] = np.nan
        metrics["spearman"] = np.nan
    else:
        metrics["pearson"] = float(pearsonr(y_true, y_pred)[0])
        metrics["spearman"] = float(spearmanr(y_true, y_pred)[0])
    return metrics


def set_backbone_trainable(model: BassetBranched, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("branched.") or name.startswith("output."):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


def load_clean_df(data_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(data_path, sep="\t").copy()
    raw_df[TARGET_COLUMN] = pd.to_numeric(raw_df[TARGET_COLUMN], errors="coerce")
    raw_df[BARCODE_COLUMN] = pd.to_numeric(raw_df[BARCODE_COLUMN], errors="coerce")
    raw_df["sequence_len"] = raw_df[SEQUENCE_COLUMN].str.len()
    clean_df = raw_df.loc[
        raw_df[SEQUENCE_COLUMN].notna()
        & raw_df[BARCODE_COLUMN].notna()
        & np.isfinite(raw_df[TARGET_COLUMN])
    ].copy().reset_index(drop=True)
    return clean_df


def split_lib1_dataset(
    df: pd.DataFrame,
    barcode_column: str = BARCODE_COLUMN,
    train_min_barcodes: int = TRAIN_MIN_BARCODES,
    high_quality_min_barcodes: int = HIGH_QUALITY_MIN_BARCODES,
    val_frac: float = VAL_FRAC_WITHIN_HIGH_QUALITY,
    test_frac: float = TEST_FRAC_WITHIN_HIGH_QUALITY,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible_df = df.loc[df[barcode_column] >= train_min_barcodes].copy()
    high_quality_df = eligible_df.loc[eligible_df[barcode_column] >= high_quality_min_barcodes].copy()
    moderate_quality_df = eligible_df.loc[
        (eligible_df[barcode_column] >= train_min_barcodes)
        & (eligible_df[barcode_column] < high_quality_min_barcodes)
    ].copy()

    rng = np.random.default_rng(random_seed)
    high_quality_indices = rng.permutation(high_quality_df.index.to_numpy())
    n_high_quality = len(high_quality_indices)
    n_test = max(1, int(round(n_high_quality * test_frac)))
    n_val = max(1, int(round(n_high_quality * val_frac)))
    if n_test + n_val >= n_high_quality:
        n_test = max(1, n_high_quality // 5)
        n_val = max(1, n_high_quality // 5)

    test_indices = high_quality_indices[:n_test]
    val_indices = high_quality_indices[n_test : n_test + n_val]
    train_high_quality_indices = high_quality_indices[n_test + n_val :]

    train_df = pd.concat(
        [
            moderate_quality_df,
            high_quality_df.loc[train_high_quality_indices],
        ],
        axis=0,
    ).sample(frac=1.0, random_state=random_seed)
    val_df = high_quality_df.loc[val_indices].sample(frac=1.0, random_state=random_seed)
    test_df = high_quality_df.loc[test_indices].sample(frac=1.0, random_state=random_seed)

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        axis=0,
    ).reset_index(drop=True)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        split_df,
    )


def prepare_split_payload(clean_df: pd.DataFrame, seed: int) -> dict[str, Any]:
    train_df, val_df, test_df, _ = split_lib1_dataset(clean_df, random_seed=seed)

    scaler = TargetScaler.from_series(train_df[TARGET_COLUMN])
    for frame in (train_df, val_df, test_df):
        frame["target_standardized"] = scaler.transform(frame[TARGET_COLUMN])
        frame["quality_band"] = np.where(
            frame[BARCODE_COLUMN] >= HIGH_QUALITY_MIN_BARCODES,
            "high_quality",
            "moderate_train_only",
        )

    split_df = pd.concat(
        [
            train_df.assign(split="train"),
            val_df.assign(split="val"),
            test_df.assign(split="test"),
        ],
        axis=0,
    ).reset_index(drop=True)

    return {
        "seed": seed,
        "scaler": {"mean": scaler.mean, "std": scaler.std},
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "split_df": split_df,
        "train_df_padded": add_padded_sequences(train_df),
        "val_df_padded": add_padded_sequences(val_df),
        "test_df_padded": add_padded_sequences(test_df),
    }


def to_scaler(payload: dict[str, Any]) -> TargetScaler:
    scaler_payload = payload["scaler"]
    return TargetScaler(mean=scaler_payload["mean"], std=scaler_payload["std"])


def train_single_head_model(
    checkpoint: dict[str, Any],
    head_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scaler: TargetScaler,
    training_seed: int,
    device: str,
    max_epochs: int = MAX_EPOCHS,
    freeze_backbone_epochs: int = FREEZE_BACKBONE_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    patience: int = EARLY_STOPPING_PATIENCE,
    batch_size: int = TRAIN_BATCH_SIZE,
    use_rc_augmentation: bool = False,
    use_barcode_weighting: bool = False,
    b_cap: float = 10.0,
    min_weight: float = 0.1,
) -> tuple[BassetBranched, pd.DataFrame, dict[str, Any]]:
    set_global_seed(training_seed)
    model = build_single_head_model(checkpoint, head_idx=head_idx, device=device, loss_criterion="MSELoss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()

    train_df_used = train_df.copy()
    if use_rc_augmentation:
        train_df_used = augment_train_df_with_rc(
            train_df_used,
            padded_column="padded_seq",
            random_seed=training_seed,
        )

    if use_barcode_weighting:
        train_df_used = add_barcode_weights(
            train_df_used,
            barcode_column=BARCODE_COLUMN,
            b_cap=b_cap,
            min_weight=min_weight,
        )
        train_loader = make_loader(
            train_df_used,
            target_column="target_standardized",
            batch_size=batch_size,
            shuffle=True,
            weight_column="sample_weight",
            seed=training_seed,
        )
    else:
        train_loader = make_loader(
            train_df_used,
            target_column="target_standardized",
            batch_size=batch_size,
            shuffle=True,
            weight_column=None,
            seed=training_seed,
        )

    val_loader = make_loader(
        val_df,
        target_column="target_standardized",
        batch_size=batch_size,
        shuffle=False,
        weight_column=None,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0
    history = []

    desc = f"seed {training_seed} | head {PRETRAINED_HEADS[head_idx]} | rc={use_rc_augmentation} | weighted={use_barcode_weighting}"
    for epoch in tqdm(range(max_epochs), desc=desc):
        set_backbone_trainable(model, trainable=epoch >= freeze_backbone_epochs)
        model.train()
        train_loss_sum = 0.0
        train_items = 0

        for batch in train_loader:
            if use_barcode_weighting:
                x_batch, y_batch, w_batch = batch
                w_batch = w_batch.to(device)
            else:
                x_batch, y_batch = batch
                w_batch = None

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_batch = model(x_batch)

            if use_barcode_weighting:
                loss = weighted_mse_loss(pred_batch, y_batch, w_batch)
            else:
                loss = criterion(pred_batch, y_batch)

            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(x_batch)
            train_items += len(x_batch)

        train_loss = train_loss_sum / max(train_items, 1)

        val_pred_std, val_true_std = predict_model(model, val_loader, device=device)
        val_pred_std = val_pred_std.reshape(-1)
        val_true_std = val_true_std.reshape(-1)
        val_loss = float(np.mean((val_pred_std - val_true_std) ** 2))

        val_pred_raw = scaler.inverse(val_pred_std)
        val_true_raw = scaler.inverse(val_true_std)
        val_metrics = compute_regression_metrics(val_true_raw, val_pred_raw)
        val_metrics.update(
            {
                "epoch": epoch,
                "train_loss_standardized": train_loss,
                "val_loss_standardized": val_loss,
                "use_rc_augmentation": use_rc_augmentation,
                "use_barcode_weighting": use_barcode_weighting,
                "b_cap": b_cap if use_barcode_weighting else np.nan,
                "training_seed": training_seed,
            }
        )
        history.append(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    fit_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss_standardized": float(best_val_loss),
        "use_rc_augmentation": use_rc_augmentation,
        "use_barcode_weighting": use_barcode_weighting,
        "b_cap": float(b_cap) if use_barcode_weighting else np.nan,
        "training_seed": int(training_seed),
    }
    return model, history_df, fit_info


def run_zero_shot_evaluation(
    checkpoint: dict[str, Any],
    test_df_padded: pd.DataFrame,
    device: str,
) -> dict[str, Any]:
    multitask_model = build_multitask_model(checkpoint, device=device)
    zero_shot_loader = make_loader(
        test_df_padded,
        target_column=TARGET_COLUMN,
        batch_size=PRED_BATCH_SIZE,
        shuffle=False,
    )
    zero_shot_pred, zero_shot_true = predict_model(multitask_model, zero_shot_loader, device=device)
    zero_shot_true = zero_shot_true.reshape(-1)

    zero_shot_results = test_df_padded[[SEQUENCE_COLUMN, BARCODE_COLUMN, TARGET_COLUMN]].copy()
    zero_shot_summary_records = []
    for head_idx, head_name in enumerate(PRETRAINED_HEADS):
        pred_column = f"pred_{head_name}"
        zero_shot_results[pred_column] = zero_shot_pred[:, head_idx]
        metrics = compute_regression_metrics(zero_shot_true, zero_shot_pred[:, head_idx])
        metrics.update({"head": head_name})
        zero_shot_summary_records.append(metrics)

    zero_shot_summary = (
        pd.DataFrame(zero_shot_summary_records)
        .sort_values("spearman", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "summary": zero_shot_summary,
        "predictions": zero_shot_results,
    }


def run_finetune_experiment(
    checkpoint: dict[str, Any],
    train_df_padded: pd.DataFrame,
    val_df_padded: pd.DataFrame,
    test_df_padded: pd.DataFrame,
    scaler: TargetScaler,
    setting: FineTuneSetting,
    run_seed: int,
    device: str,
) -> dict[str, Any]:
    fine_tune_runs = {}
    fine_tune_summary_records = []

    for head_idx, head_name in enumerate(PRETRAINED_HEADS):
        training_seed = run_seed * 100 + head_idx
        model, history_df, fit_info = train_single_head_model(
            checkpoint=checkpoint,
            head_idx=head_idx,
            train_df=train_df_padded,
            val_df=val_df_padded,
            scaler=scaler,
            training_seed=training_seed,
            device=device,
            use_rc_augmentation=setting.use_rc_augmentation,
            use_barcode_weighting=setting.use_barcode_weighting,
            b_cap=float(setting.b_cap if setting.b_cap is not None else 10.0),
            min_weight=setting.min_weight,
        )

        test_loader = make_loader(
            test_df_padded,
            target_column="target_standardized",
            batch_size=PRED_BATCH_SIZE,
            shuffle=False,
        )
        test_pred_std, test_true_std = predict_model(model, test_loader, device=device)
        test_pred_std = test_pred_std.reshape(-1)
        test_true_std = test_true_std.reshape(-1)

        test_pred_raw = scaler.inverse(test_pred_std)
        test_true_raw = scaler.inverse(test_true_std)
        test_metrics = compute_regression_metrics(test_true_raw, test_pred_raw)
        test_metrics.update(
            {
                "init_head": head_name,
                "run_seed": int(run_seed),
                "setting": setting.name,
                **fit_info,
            }
        )
        fine_tune_summary_records.append(test_metrics)

        run_df = test_df_padded[[SEQUENCE_COLUMN, BARCODE_COLUMN, TARGET_COLUMN]].copy()
        run_df["pred_finetuned"] = test_pred_raw

        fine_tune_runs[head_name] = {
            "history": history_df,
            "test_predictions": run_df,
            "fit_info": fit_info,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fine_tune_summary = (
        pd.DataFrame(fine_tune_summary_records)
        .sort_values("spearman", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "summary": fine_tune_summary,
        "runs": fine_tune_runs,
    }


def save_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def maybe_load_or_build(cache_path: Path, force: bool, builder) -> Any:
    if cache_path.exists() and not force:
        return load_pickle(cache_path)
    payload = builder()
    save_pickle(cache_path, payload)
    return payload


def build_settings(b3_bcap: float, min_weight: float) -> list[FineTuneSetting]:
    return [
        FineTuneSetting(name="B1_no_RC", use_rc_augmentation=False, use_barcode_weighting=False),
        FineTuneSetting(name="B2_with_RC", use_rc_augmentation=True, use_barcode_weighting=False),
        FineTuneSetting(
            name=f"B3_with_RC_weighted_bcap_{b3_bcap:g}",
            use_rc_augmentation=True,
            use_barcode_weighting=True,
            b_cap=b3_bcap,
            min_weight=min_weight,
        ),
    ]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    flat_columns = []
    for column in frame.columns:
        if isinstance(column, tuple):
            flat_columns.append("_".join(str(part) for part in column if part).rstrip("_"))
        else:
            flat_columns.append(str(column))
    frame.columns = flat_columns
    return frame


def aggregate_metric_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    summary = frame.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    summary = flatten_columns(summary)
    return summary.sort_values(group_cols).reset_index(drop=True)


def ensure_cache_dirs(cache_root: Path) -> dict[str, Path]:
    paths = {
        "root": cache_root,
        "datasets": cache_root / "datasets",
        "zero_shot": cache_root / "zero_shot",
        "runs": cache_root / "runs",
        "aggregates": cache_root / "aggregates",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_manifest(
    manifest_path: Path,
    args: argparse.Namespace,
    settings: list[FineTuneSetting],
    device: str,
) -> None:
    manifest = {
        "repo_root": str(REPO_ROOT),
        "data_path": str(args.data_path),
        "model_path": str(args.model_path),
        "cache_root": str(args.cache_root),
        "device": device,
        "seeds": list(args.seeds),
        "settings": [asdict(setting) for setting in settings],
        "train_min_barcodes": TRAIN_MIN_BARCODES,
        "high_quality_min_barcodes": HIGH_QUALITY_MIN_BARCODES,
        "val_frac_within_high_quality": VAL_FRAC_WITHIN_HIGH_QUALITY,
        "test_frac_within_high_quality": TEST_FRAC_WITHIN_HIGH_QUALITY,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cached multi-seed B1/B2/B3 fine-tuning on lib1 enhancer data."
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--b3_bcap", type=float, default=8.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Ignore caches and recompute outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    device = resolve_device(args.device)
    cache_dirs = ensure_cache_dirs(args.cache_root)
    settings = build_settings(b3_bcap=args.b3_bcap, min_weight=args.min_weight)
    write_manifest(cache_dirs["root"] / "run_manifest.json", args, settings, device)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Cache root: {args.cache_root}")
    print(f"Seeds: {args.seeds}")

    clean_df = load_clean_df(args.data_path)
    checkpoint = load_checkpoint_from_tar(args.model_path, map_location="cpu")

    zero_shot_by_seed = []
    fine_tune_by_seed = []

    for seed in args.seeds:
        print(f"\n=== Seed {seed} ===")
        split_cache_path = cache_dirs["datasets"] / f"split_seed_{seed}.pkl"
        split_payload = maybe_load_or_build(
            split_cache_path,
            force=args.force,
            builder=lambda current_seed=seed: prepare_split_payload(clean_df, current_seed),
        )
        scaler = to_scaler(split_payload)

        zero_shot_cache_path = cache_dirs["zero_shot"] / f"seed_{seed}.pkl"
        zero_shot_payload = maybe_load_or_build(
            zero_shot_cache_path,
            force=args.force,
            builder=lambda: run_zero_shot_evaluation(
                checkpoint=checkpoint,
                test_df_padded=split_payload["test_df_padded"],
                device=device,
            ),
        )
        zero_shot_summary = zero_shot_payload["summary"].copy()
        zero_shot_summary["seed"] = seed
        zero_shot_by_seed.append(zero_shot_summary)

        for setting in settings:
            setting_dir = cache_dirs["runs"] / setting.cache_name
            setting_cache_path = setting_dir / f"seed_{seed}.pkl"
            setting_payload = maybe_load_or_build(
                setting_cache_path,
                force=args.force,
                builder=lambda current_setting=setting, current_seed=seed: run_finetune_experiment(
                    checkpoint=checkpoint,
                    train_df_padded=split_payload["train_df_padded"],
                    val_df_padded=split_payload["val_df_padded"],
                    test_df_padded=split_payload["test_df_padded"],
                    scaler=scaler,
                    setting=current_setting,
                    run_seed=current_seed,
                    device=device,
                ),
            )
            setting_summary = setting_payload["summary"].copy()
            setting_summary["seed"] = seed
            fine_tune_by_seed.append(setting_summary)

    zero_shot_by_seed_df = pd.concat(zero_shot_by_seed, ignore_index=True)
    fine_tune_by_seed_df = pd.concat(fine_tune_by_seed, ignore_index=True)

    zero_shot_aggregate_df = aggregate_metric_summary(
        zero_shot_by_seed_df,
        group_cols=["head"],
        metrics=["pearson", "spearman", "rmse", "mae"],
    )
    fine_tune_aggregate_df = aggregate_metric_summary(
        fine_tune_by_seed_df,
        group_cols=["setting", "init_head"],
        metrics=["pearson", "spearman", "rmse", "mae", "best_epoch", "best_val_loss_standardized"],
    )

    zero_shot_by_seed_df.to_csv(cache_dirs["aggregates"] / "zero_shot_summary_by_seed.csv", index=False)
    fine_tune_by_seed_df.to_csv(cache_dirs["aggregates"] / "finetune_summary_by_seed.csv", index=False)
    zero_shot_aggregate_df.to_csv(cache_dirs["aggregates"] / "zero_shot_summary_mean_std.csv", index=False)
    fine_tune_aggregate_df.to_csv(cache_dirs["aggregates"] / "finetune_summary_mean_std.csv", index=False)

    plot_ready_zero_shot_df = zero_shot_aggregate_df.rename(columns={"head": "init_head"}).assign(setting="zero_shot")
    plot_ready_df = pd.concat([plot_ready_zero_shot_df, fine_tune_aggregate_df], ignore_index=True, sort=False)
    plot_ready_df.to_csv(cache_dirs["aggregates"] / "barplot_summary_mean_std.csv", index=False)

    print("\nWrote aggregate outputs:")
    print(f"  {cache_dirs['aggregates'] / 'zero_shot_summary_by_seed.csv'}")
    print(f"  {cache_dirs['aggregates'] / 'finetune_summary_by_seed.csv'}")
    print(f"  {cache_dirs['aggregates'] / 'zero_shot_summary_mean_std.csv'}")
    print(f"  {cache_dirs['aggregates'] / 'finetune_summary_mean_std.csv'}")
    print(f"  {cache_dirs['aggregates'] / 'barplot_summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
