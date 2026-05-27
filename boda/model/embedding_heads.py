import argparse

import torch
import torch.nn as nn

from .loss_functions import add_criterion_specific_args


def _activation(name):
    if not hasattr(nn, name):
        raise ValueError(f"Unknown torch.nn activation: {name}")
    return getattr(nn, name)()


def _linear(in_features, out_features):
    if in_features is None or int(in_features) <= 0:
        return nn.LazyLinear(out_features)
    return nn.Linear(int(in_features), out_features)


class EmbeddingMLPRegressor(nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group("Model Module args")
        group.add_argument("--input_dim", type=int, default=0)
        group.add_argument("--n_outputs", type=int, default=1)
        group.add_argument("--hidden_dim", type=int, default=512)
        group.add_argument("--n_hidden_layers", type=int, default=2)
        group.add_argument("--dropout_p", type=float, default=0.1)
        group.add_argument("--activation", type=str, default="GELU")
        group.add_argument("--loss_criterion", type=str, default="MSELoss")
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_criterion_specific_args(parser, known_args.loss_criterion)
        return parser

    @staticmethod
    def process_args(grouped_args):
        model_args = grouped_args["Model Module args"]
        model_args.loss_args = vars(grouped_args["Criterion args"])
        return model_args

    def __init__(
        self,
        input_dim=0,
        n_outputs=1,
        hidden_dim=512,
        n_hidden_layers=2,
        dropout_p=0.1,
        activation="GELU",
        loss_criterion="MSELoss",
        loss_args=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_outputs = n_outputs
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.dropout_p = dropout_p
        self.activation = activation
        self.loss_criterion = loss_criterion
        self.loss_args = {} if loss_args is None else loss_args

        layers = []
        if n_hidden_layers <= 0:
            layers.append(_linear(input_dim, n_outputs))
        else:
            layers.append(_linear(input_dim, hidden_dim))
            layers.append(_activation(activation))
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(_activation(activation))
                if dropout_p > 0:
                    layers.append(nn.Dropout(dropout_p))
            layers.append(nn.Linear(hidden_dim, n_outputs))
        self.network = nn.Sequential(*layers)
        self.criterion = getattr(nn, loss_criterion)(**self.loss_args)

    def forward(self, x):
        return self.network(x)


class EmbeddingHeteroscedasticRegressor(EmbeddingMLPRegressor):
    def __init__(
        self,
        input_dim=0,
        n_outputs=1,
        hidden_dim=512,
        n_hidden_layers=2,
        dropout_p=0.1,
        activation="GELU",
        loss_criterion="MSELoss",
        loss_args=None,
    ):
        n_targets = int(n_outputs)
        super().__init__(
            input_dim=input_dim,
            n_outputs=n_targets * 2,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            dropout_p=dropout_p,
            activation=activation,
            loss_criterion=loss_criterion,
            loss_args=loss_args,
        )
        self.n_targets = n_targets

    def split_mu_log_var(self, output):
        mu, log_var = torch.split(output, self.n_targets, dim=-1)
        return mu, torch.clamp(log_var, min=-10.0, max=5.0)
