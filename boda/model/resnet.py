import argparse
import torch
import torch.nn as nn
import lightning.pytorch as ptl

from ..common import utils
from .loss_functions import add_criterion_specific_args
from ..model import loss_functions


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout_p=0.0, use_batch_norm=True):
        super().__init__()
        pad = kernel_size // 2
        norm = nn.BatchNorm1d if use_batch_norm else nn.Identity

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=not use_batch_norm)
        self.bn1 = norm(out_channels) if use_batch_norm else nn.Identity()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=pad, bias=not use_batch_norm)
        self.bn2 = norm(out_channels) if use_batch_norm else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm(out_channels) if use_batch_norm else nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class ResNet1DRegressor(ptl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Model Module args')
        group.add_argument('--input_len', type=int, default=600)
        group.add_argument('--stem_channels', type=int, default=64)
        group.add_argument('--stem_kernel_size', type=int, default=15)
        group.add_argument('--stage_channels', type=int, nargs='+', default=[64, 128, 256])
        group.add_argument('--stage_blocks', type=int, nargs='+', default=[2, 2, 2])
        group.add_argument('--block_kernel_size', type=int, default=7)
        group.add_argument('--dropout_p', type=float, default=0.2)
        group.add_argument('--head_hidden_channels', type=int, default=128)
        group.add_argument('--n_outputs', type=int, default=1)
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True)
        group.add_argument('--loss_criterion', type=str, default='MSELoss')
        return parser

    @staticmethod
    def add_conditional_args(parser, known_args):
        parser = add_criterion_specific_args(parser, known_args.loss_criterion)
        return parser

    @staticmethod
    def process_args(grouped_args):
        model_args = grouped_args['Model Module args']
        model_args.loss_args = vars(grouped_args['Criterion args'])
        return model_args

    def __init__(self,
                 input_len=600,
                 stem_channels=64,
                 stem_kernel_size=15,
                 stage_channels=(64, 128, 256),
                 stage_blocks=(2, 2, 2),
                 block_kernel_size=7,
                 dropout_p=0.2,
                 head_hidden_channels=128,
                 n_outputs=1,
                 use_batch_norm=True,
                 loss_criterion='MSELoss',
                 loss_args={}):
        super().__init__()
        self.input_len = input_len
        self.n_outputs = n_outputs
        self.loss_criterion = loss_criterion
        self.loss_args = loss_args

        stem_pad = stem_kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_channels, stem_kernel_size, stride=1, padding=stem_pad, bias=not use_batch_norm),
            nn.BatchNorm1d(stem_channels) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
        )

        channels = [stem_channels] + list(stage_channels)
        blocks = []
        for stage_idx, (in_ch, out_ch, n_block) in enumerate(zip(channels[:-1], channels[1:], stage_blocks)):
            for block_idx in range(n_block):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                block_in = in_ch if block_idx == 0 else out_ch
                blocks.append(
                    ResidualBlock1D(
                        in_channels=block_in,
                        out_channels=out_ch,
                        kernel_size=block_kernel_size,
                        stride=stride,
                        dropout_p=dropout_p,
                        use_batch_norm=use_batch_norm,
                    )
                )
        self.encoder = nn.Sequential(*blocks)
        final_channels = list(stage_channels)[-1] if len(stage_channels) > 0 else stem_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, head_hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(head_hidden_channels, n_outputs),
        )
        self.criterion = getattr(loss_functions, self.loss_criterion)(**self.loss_args)

    def encode(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        return self.head(self.encode(x))
