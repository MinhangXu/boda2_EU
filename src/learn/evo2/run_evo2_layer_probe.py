#!/usr/bin/env python
import argparse
import glob
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boda.data import EmbeddingRegressionDataModule
from boda.graph import EmbeddingRegressionTraining, WeightedEmbeddingRegressionTraining
from boda.model import EmbeddingMLPRegressor


def find_embedding_files(embedding_dir, embedding_files, embedding_glob):
    if embedding_files:
        files = []
        for path in embedding_files:
            candidate = Path(path)
            if not candidate.exists() and not candidate.is_absolute():
                candidate = Path(embedding_dir) / candidate
            files.append(candidate.resolve())
    else:
        files = [Path(path).resolve() for path in sorted(glob.glob(str(Path(embedding_dir) / embedding_glob)))]
    if not files:
        raise ValueError(f"No embedding files found in {embedding_dir} with pattern {embedding_glob}")
    return files


def scalar_metric(metrics, key):
    value = metrics.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def layer_name_from_file(path):
    payload = torch.load(path, map_location="cpu")
    return payload.get("layer_name", path.stem), int(payload["embedding"].shape[1])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run fixed-head probes across cached Evo2 embedding layers.")
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--embedding_files", nargs="*", default=None)
    parser.add_argument("--embedding_glob", default="embeddings__*.pt")
    parser.add_argument("--rows_file", default="rows.parquet")
    parser.add_argument("--target_columns", nargs="+", required=True)
    parser.add_argument("--split_column", default="split")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--standardize_x", action="store_true", default=True)
    parser.add_argument("--no_standardize_x", dest="standardize_x", action="store_false")
    parser.add_argument("--standardize_y", action="store_true", default=True)
    parser.add_argument("--no_standardize_y", dest="standardize_y", action="store_false")
    parser.add_argument("--use_weights", action="store_true")
    parser.add_argument("--weight_column", default="n_barcodes")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_hidden_layers", type=int, default=1)
    parser.add_argument("--dropout_p", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint_monitor", default="val_pearson_mean")
    parser.add_argument("--stopping_mode", choices=["min", "max"], default="max")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--output_csv", default="layer_probe_summary.csv")
    parser.add_argument("--default_root_dir", default="src/learn/local_artifacts/foundation_layer_probe")
    parser.add_argument("--include_test_metrics", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if isinstance(args.devices, str) and args.devices.isdigit():
        args.devices = int(args.devices)
    torch.manual_seed(args.seed)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_root = Path(args.default_root_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    graph_cls = WeightedEmbeddingRegressionTraining if args.use_weights else EmbeddingRegressionTraining
    for embedding_file in find_embedding_files(args.embedding_dir, args.embedding_files, args.embedding_glob):
        layer_name, embedding_dim = layer_name_from_file(embedding_file)
        safe_layer = str(layer_name).replace("/", "_").replace(".", "_")
        data = EmbeddingRegressionDataModule(
            embedding_dir=args.embedding_dir,
            embedding_file=str(embedding_file),
            rows_file=args.rows_file,
            target_columns=args.target_columns,
            split_column=args.split_column,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            standardize_x=args.standardize_x,
            standardize_y=args.standardize_y,
            use_weights=args.use_weights,
            weight_column=args.weight_column,
        )
        data.setup("fit")
        model = EmbeddingMLPRegressor(
            input_dim=data.input_dim,
            n_outputs=data.n_outputs,
            hidden_dim=args.hidden_dim,
            n_hidden_layers=args.n_hidden_layers,
            dropout_p=args.dropout_p,
            loss_criterion="MSELoss",
        )
        graph = graph_cls(
            model=model,
            optimizer="AdamW",
            optimizer_args={"lr": args.lr},
            output_names=args.target_columns,
        )
        logger = CSVLogger(save_dir=str(run_root), name=f"layer_{safe_layer}")
        checkpoint = ModelCheckpoint(
            save_top_k=1,
            monitor=args.checkpoint_monitor,
            mode=args.stopping_mode,
        )
        early_stopping = EarlyStopping(
            monitor=args.checkpoint_monitor,
            patience=args.patience,
            mode=args.stopping_mode,
        )
        trainer = Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=logger,
            callbacks=[checkpoint, early_stopping],
            default_root_dir=str(run_root / f"layer_{safe_layer}"),
            enable_progress_bar=True,
        )
        trainer.fit(graph, datamodule=data)
        metrics = dict(trainer.callback_metrics)
        try:
            embedding_file_label = os.path.relpath(embedding_file, Path(args.embedding_dir).resolve())
        except ValueError:
            embedding_file_label = str(embedding_file)
        record = {
            "layer_name": layer_name,
            "embedding_file": embedding_file_label,
            "embedding_dim": embedding_dim,
            "n_train": data.split_summary["n_train"],
            "n_val": data.split_summary["n_val"],
            "n_test": data.split_summary["n_test"],
            "target_columns": " ".join(args.target_columns),
            "best_checkpoint": checkpoint.best_model_path,
            "best_model_score": scalar_metric({"score": checkpoint.best_model_score}, "score"),
        }
        for key in [
            "val_loss",
            "val_mse",
            "val_pearson_mean",
            "val_cod_r2_mean",
            "val_spearman_mean",
        ]:
            record[key] = scalar_metric(metrics, key)

        if args.include_test_metrics and checkpoint.best_model_path:
            best = graph_cls(
                model=model,
                optimizer="AdamW",
                optimizer_args={"lr": args.lr},
                output_names=args.target_columns,
            )
            state = torch.load(checkpoint.best_model_path, map_location="cpu")
            best.load_state_dict(state["state_dict"])
            trainer.test(best, datamodule=data)
            test_metrics = dict(trainer.callback_metrics)
            for key in ["test_loss", "test_pearson_mean", "test_cod_r2_mean", "test_spearman_mean"]:
                record[key] = scalar_metric(test_metrics, key)
        rows.append(record)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_csv, index=False)
    print(summary.sort_values("val_pearson_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
