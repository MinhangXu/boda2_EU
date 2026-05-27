import argparse
import ast

import torch
from lightning.pytorch import LightningModule

import hypertune

from .utils import (
    add_optimizer_specific_args,
    add_scheduler_specific_args,
    coefficient_of_determination,
    normalize_scheduler_name,
    pearson_correlation,
    pearson_r2_score,
    reorg_optimizer_args,
    reorg_scheduler_args,
    spearman_correlation,
)


def _coerce_optional_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "null"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
        return value.split()
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            coerced = _coerce_optional_list(item)
            if coerced is not None:
                flattened.extend(coerced)
        return flattened
    return [str(value)]


class EmbeddingRegressionTraining(LightningModule):
    @staticmethod
    def add_graph_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("Graph Module args")
        group.add_argument("--optimizer", type=str, default="Adam")
        group.add_argument("--scheduler", type=str)
        group.add_argument("--scheduler_monitor", type=str)
        group.add_argument("--scheduler_interval", type=str, default="epoch")
        group.add_argument("--output_names", type=str, nargs="+", default=None)
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_optimizer_specific_args(parser, known_args.optimizer)
        parser = add_scheduler_specific_args(parser, known_args.scheduler)
        return parser

    @staticmethod
    def process_args(grouped_args):
        graph_args = grouped_args["Graph Module args"]
        graph_args.output_names = _coerce_optional_list(graph_args.output_names)
        graph_args.scheduler = normalize_scheduler_name(graph_args.scheduler)
        graph_args.optimizer_args = vars(grouped_args["Optimizer args"])
        graph_args.optimizer_args = reorg_optimizer_args(graph_args.optimizer_args)
        try:
            graph_args.scheduler_args = vars(grouped_args["LR Scheduler args"])
            graph_args.scheduler_args = reorg_scheduler_args(graph_args.scheduler_args)
        except KeyError:
            graph_args.scheduler_args = None
        return graph_args

    def __init__(
        self,
        model,
        optimizer="Adam",
        scheduler=None,
        scheduler_monitor=None,
        scheduler_interval="epoch",
        optimizer_args=None,
        scheduler_args=None,
        output_names=None,
    ):
        super().__init__()
        self.model = model
        self.criterion = model.criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval
        self.optimizer_args = optimizer_args or {}
        self.scheduler_args = scheduler_args
        self.output_names = _coerce_optional_list(output_names)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.hpt = hypertune.HyperTune()
        params = [p for p in self.parameters() if p.requires_grad]
        try:
            n_params = sum(p.numel() for p in params)
        except ValueError:
            n_params = "uninitialized lazy"
        print(f"Found {n_params} parameters")
        optimizer_class = getattr(torch.optim, self.optimizer)
        optimizer = optimizer_class(self.parameters(), **self.optimizer_args)
        scheduler_name = normalize_scheduler_name(self.scheduler)
        if scheduler_name is None:
            return optimizer
        scheduler_dict = {
            "scheduler": getattr(torch.optim.lr_scheduler, scheduler_name)(
                optimizer, **(self.scheduler_args or {})
            ),
            "interval": self.scheduler_interval,
            "name": "learning_rate",
        }
        if self.scheduler_monitor is not None:
            scheduler_dict["monitor"] = self.scheduler_monitor
        return [optimizer], [scheduler_dict]

    def _unpack_batch(self, batch):
        if len(batch) == 2:
            x, y = batch
            weight = None
        elif len(batch) == 3:
            x, y, weight = batch
        else:
            raise ValueError(f"Unexpected embedding batch length: {len(batch)}")
        return x, y, weight

    def _align_shapes(self, y_hat, y):
        if y_hat.dim() == 1:
            y_hat = y_hat.view(-1, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)
        return y_hat, y

    def _prediction_from_output(self, output):
        return output

    def _loss(self, output, y, weight=None):
        y_hat, y = self._align_shapes(self._prediction_from_output(output), y)
        return self.criterion(y_hat, y)

    def _output_names_for(self, n_outputs):
        if self.output_names is not None and len(self.output_names) == n_outputs:
            return self.output_names
        if n_outputs == 1:
            return ["target"]
        return [f"target_{i}" for i in range(n_outputs)]

    def _log_per_output(self, prefix, pearson_vals, spearman_vals, mse_vals, **kwargs):
        names = self._output_names_for(int(pearson_vals.numel()))
        for idx, name in enumerate(names):
            self.log(f"{prefix}_pearson_{name}", pearson_vals[idx], **kwargs)
            self.log(f"{prefix}_pearson_squared_{name}", pearson_vals[idx] ** 2, **kwargs)
            self.log(f"{prefix}_spearman_{name}", spearman_vals[idx], **kwargs)
            self.log(f"{prefix}_mse_{name}", mse_vals[idx], **kwargs)

    def _epoch_metrics(self, prefix, outputs):
        if not outputs:
            return None
        loss = torch.stack([batch["loss"] for batch in outputs], dim=0).mean()
        y_hat = torch.cat([batch["preds"] for batch in outputs], dim=0)
        y = torch.cat([batch["labels"] for batch in outputs], dim=0)
        y_hat, y = self._align_shapes(y_hat, y)

        mse_vals = (y_hat - y).pow(2).mean(dim=0)
        mse_mean = mse_vals.mean()
        pearson_vals, pearson_mean = pearson_correlation(y_hat, y)
        spearman_vals, spearman_mean = spearman_correlation(y_hat, y)
        pearson_r2 = pearson_r2_score(y, y_hat)
        cod_r2 = coefficient_of_determination(y, y_hat)

        on_epoch = True
        self.log(f"{prefix}_loss", loss, on_epoch=on_epoch)
        self.log(f"{prefix}_mse", mse_mean, on_epoch=on_epoch)
        self.log(f"{prefix}_standardized_mse", mse_mean, on_epoch=on_epoch)
        self.log(f"{prefix}_pearson", pearson_mean, on_epoch=on_epoch)
        self.log(f"{prefix}_pearson_mean", pearson_mean, on_epoch=on_epoch)
        self.log(f"{prefix}_pearson_r2", pearson_r2, on_epoch=on_epoch)
        self.log(f"{prefix}_cod_r2", cod_r2, on_epoch=on_epoch)
        self.log(f"{prefix}_cod_r2_mean", cod_r2, on_epoch=on_epoch)
        self.log(f"{prefix}_spearman", spearman_mean, on_epoch=on_epoch)
        self.log(f"{prefix}_spearman_mean", spearman_mean, on_epoch=on_epoch)
        self._log_per_output(prefix, pearson_vals, spearman_vals, mse_vals, on_epoch=on_epoch)
        if prefix == "val":
            self.log("valid_loss", loss, on_epoch=on_epoch)
        return None

    def training_step(self, batch, batch_idx):
        x, y, weight = self._unpack_batch(batch)
        output = self(x)
        loss = self._loss(output, y, weight=None)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, weight = self._unpack_batch(batch)
        output = self(x)
        loss = self._loss(output, y, weight=None)
        y_hat, y = self._align_shapes(self._prediction_from_output(output), y)
        self.log("step_valid_loss", loss)
        self.log("valid_loss", loss)
        return {"loss": loss.detach(), "preds": y_hat.detach(), "labels": y.detach()}

    def validation_epoch_end(self, val_step_outputs):
        return self._epoch_metrics("val", val_step_outputs)

    def test_step(self, batch, batch_idx):
        x, y, weight = self._unpack_batch(batch)
        output = self(x)
        loss = self._loss(output, y, weight=None)
        y_hat, y = self._align_shapes(self._prediction_from_output(output), y)
        self.log("step_test_loss", loss, on_epoch=True)
        return {"loss": loss.detach(), "preds": y_hat.detach(), "labels": y.detach()}

    def test_epoch_end(self, test_step_outputs):
        return self._epoch_metrics("test", test_step_outputs)


class WeightedEmbeddingRegressionTraining(EmbeddingRegressionTraining):
    def _loss(self, output, y, weight=None):
        y_hat, y = self._align_shapes(self._prediction_from_output(output), y)
        if weight is None:
            return self.criterion(y_hat, y)
        if weight.dim() > 1:
            weight = weight.reshape(-1)
        per_sample = (y_hat - y).pow(2)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        weight = weight.to(per_sample.dtype)
        return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)

    def training_step(self, batch, batch_idx):
        x, y, weight = self._unpack_batch(batch)
        output = self(x)
        loss = self._loss(output, y, weight=weight)
        self.log("train_loss", loss)
        return loss


class HeteroscedasticEmbeddingRegressionTraining(EmbeddingRegressionTraining):
    def _prediction_from_output(self, output):
        if hasattr(self.model, "split_mu_log_var"):
            mu, _ = self.model.split_mu_log_var(output)
            return mu
        n_targets = output.shape[-1] // 2
        return output[..., :n_targets]

    def _split_output(self, output):
        if hasattr(self.model, "split_mu_log_var"):
            return self.model.split_mu_log_var(output)
        n_targets = output.shape[-1] // 2
        mu, log_var = torch.split(output, n_targets, dim=-1)
        return mu, torch.clamp(log_var, min=-10.0, max=5.0)

    def _loss(self, output, y, weight=None):
        mu, log_var = self._split_output(output)
        mu, y = self._align_shapes(mu, y)
        per_output = 0.5 * (torch.exp(-log_var) * (y - mu).pow(2) + log_var)
        per_sample = per_output.mean(dim=1)
        if weight is not None:
            if weight.dim() > 1:
                weight = weight.reshape(-1)
            weight = weight.to(per_sample.dtype)
            return (per_sample * weight).sum() / weight.sum().clamp_min(1e-8)
        return per_sample.mean()

    def training_step(self, batch, batch_idx):
        x, y, weight = self._unpack_batch(batch)
        output = self(x)
        loss = self._loss(output, y, weight=weight)
        self.log("train_loss", loss)
        return loss
