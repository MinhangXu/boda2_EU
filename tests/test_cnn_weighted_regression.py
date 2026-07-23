import argparse

import pytest
import torch
from torch import nn

from boda.graph import CNNWeightedRegressionTraining


class IdentityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return x[:, :1]


class TwoOutputIdentityRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return x[:, :2]


def test_weighted_graph_accepts_basic_logging_args():
    parser = CNNWeightedRegressionTraining.add_graph_specific_args(
        argparse.ArgumentParser(add_help=False)
    )
    args = parser.parse_args(
        [
            "--log_legacy_metric_aliases",
            "false",
            "--log_per_output_metric_details",
            "false",
            "--weighted_loss_reduction",
            "mean",
        ]
    )

    assert args.log_legacy_metric_aliases is False
    assert args.log_per_output_metric_details is False
    assert args.weighted_loss_reduction == "mean"


def test_weighted_training_loss_and_unweighted_validation_loss():
    graph = CNNWeightedRegressionTraining(
        model=IdentityRegressor(),
        optimizer_args={"lr": 1e-3},
        log_legacy_metric_aliases=False,
    )
    x = torch.tensor([[0.0], [2.0], [4.0]])
    y = torch.tensor([[1.0], [1.0], [1.0]])
    weights = torch.tensor([1.0, 0.5, 0.25])

    train_out = graph.training_step((x, y, weights), 0)
    expected_weighted = (((x - y).pow(2).view(-1) * weights).sum() / weights.sum())
    assert torch.allclose(train_out["loss"], expected_weighted)
    assert set(train_out) == {"loss", "preds", "labels"}

    val_out = graph.validation_step((x, y, weights), 0)
    expected_unweighted = nn.MSELoss()(x, y)
    assert torch.allclose(val_out["loss"], expected_unweighted)


def test_weighted_training_averages_outputs_before_weighting_samples():
    graph = CNNWeightedRegressionTraining(
        model=TwoOutputIdentityRegressor(),
        optimizer_args={"lr": 1e-3},
        log_legacy_metric_aliases=False,
    )
    x = torch.tensor([[0.0, 2.0], [3.0, 5.0]])
    y = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    weights = torch.tensor([1.0, 3.0])

    per_sample = (x - y).pow(2).mean(dim=1)
    expected = (per_sample * weights).sum() / weights.sum()
    observed = graph.training_step((x, y, weights), 0)["loss"]

    assert torch.allclose(observed, expected)


def test_weighted_process_args_propagates_reduction():
    grouped = {
        "Graph Module args": argparse.Namespace(
            optimizer="Adam",
            scheduler=None,
            scheduler_monitor=None,
            scheduler_interval="epoch",
            output_names=None,
            log_per_output_metric_details=False,
            log_legacy_metric_aliases=False,
        ),
        "Optimizer args": argparse.Namespace(lr=1e-3),
        "Weighted regression args": argparse.Namespace(weighted_loss_reduction="mean"),
    }

    processed = CNNWeightedRegressionTraining.process_args(grouped)

    assert processed.weighted_loss_reduction == "mean"


def test_weighted_training_fails_closed_when_weights_are_missing():
    graph = CNNWeightedRegressionTraining(
        model=IdentityRegressor(),
        optimizer_args={"lr": 1e-3},
        log_legacy_metric_aliases=False,
    )
    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[1.0], [1.0]])

    with pytest.raises(ValueError, match="requires a three-item"):
        graph.training_step((x, y), 0)


@pytest.mark.parametrize(
    "weights, message",
    [
        (torch.tensor([1.0]), "one weight per sample"),
        (torch.tensor([1.0, float("nan")]), "must all be finite"),
        (torch.tensor([1.0, -0.1]), "must be nonnegative"),
        (torch.tensor([0.0, 0.0]), "positive finite sum"),
    ],
)
def test_weighted_training_rejects_invalid_weights(weights, message):
    graph = CNNWeightedRegressionTraining(
        model=IdentityRegressor(),
        optimizer_args={"lr": 1e-3},
        log_legacy_metric_aliases=False,
    )
    x = torch.tensor([[0.0], [2.0]])
    y = torch.tensor([[1.0], [1.0]])

    with pytest.raises(ValueError, match=message):
        graph.training_step((x, y, weights), 0)
