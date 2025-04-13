import argparse
import sys
import math
from collections import OrderedDict

import torch
import torch.nn as nn

import lightning.pytorch as ptl

from ..common import utils
from .loss_functions import add_criterion_specific_args

from ..model import loss_functions

class ResidualBlock(nn.Module):
    """
    A single residual block as used in ResNet architectures.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int, optional): Stride of the convolution. Default is 1.
        dropout_p (float, optional): Dropout probability. Default is 0.1.
        activation (str, optional): Activation function name. Default is 'LeakyReLU'.
        use_batch_norm (bool, optional): Whether to use batch normalization. Default is True.
        use_weight_norm (bool, optional): Whether to use weight normalization. Default is False.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 dropout_p=0.1, activation='LeakyReLU', 
                 use_batch_norm=True, use_weight_norm=False):
        super().__init__()
        
        # Determine padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        
        # First convolutional block
        conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding)
        if use_weight_norm:
            conv1 = nn.utils.weight_norm(conv1)
            
        blocks = [conv1]
        
        if use_batch_norm:
            blocks.append(nn.BatchNorm1d(out_channels))
            
        blocks.append(getattr(nn, activation)())
        blocks.append(nn.Dropout(dropout_p))
        
        # Second convolutional block
        conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                         stride=1, padding=padding)
        if use_weight_norm:
            conv2 = nn.utils.weight_norm(conv2)
            
        blocks.extend([
            conv2,
            nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity(),
            getattr(nn, activation)(),
            nn.Dropout(dropout_p)
        ])
        
        self.conv_stack = nn.Sequential(*blocks)
        
        # Skip connection projection if needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        """
        Forward pass through the residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv_stack(x) + self.shortcut(x)


class ResNetModule(ptl.LightningModule):
    """
    ResNet model architecture for genomic sequences.
    
    Args:
        input_len (int): Fixed sequence length of inputs.
        n_res_blocks (int): Number of residual blocks.
        filters_per_block (list): Number of filters for each residual block.
        kernel_sizes (list): Kernel sizes for each residual block.
        stride_sizes (list, optional): Stride sizes for each residual block.
        fc_layers (list): Number of units in fully connected layers.
        n_outputs (int): Number of output units.
        conv_dropout_p (float): Dropout probability for convolutional layers.
        fc_dropout_p (float): Dropout probability for fully connected layers.
        activation (str): Activation function name.
        use_batch_norm (bool): Whether to use batch normalization.
        use_weight_norm (bool): Whether to use weight normalization.
        loss_criterion (str): Loss criterion name.
        loss_args (dict): Arguments for the loss criterion.
    """
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Add model-specific arguments to the provided argument parser.
        
        Args:
            parent_parser (argparse.ArgumentParser): The parent argument parser.
            
        Returns:
            argparse.ArgumentParser: The argument parser with added model-specific arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Model Module args')
        
        group.add_argument('--input_len', type=int, default=84,
                          help='Fixed sequence length of inputs')
        
        # ResNet architecture params
        group.add_argument('--n_res_blocks', type=int, default=3, 
                          help='Number of residual blocks')
        group.add_argument('--filters_per_block', type=int, nargs='+', default=[64, 128, 256],
                          help='Number of filters for each residual block')
        group.add_argument('--kernel_sizes', type=int, nargs='+', default=[3, 3, 3],
                          help='Kernel sizes for each residual block')
        group.add_argument('--stride_sizes', type=int, nargs='+', default=[1, 1, 1],
                          help='Stride sizes for each residual block')
        
        # Fully connected layers params
        group.add_argument('--fc_layers', type=int, nargs='+', default=[128, 64],
                          help='Number of units in fully connected layers')
        group.add_argument('--n_outputs', type=int, default=1,
                          help='Number of output units')
        
        # Regularization params
        group.add_argument('--conv_dropout_p', type=float, default=0.1,
                          help='Dropout probability for convolutional layers')
        group.add_argument('--fc_dropout_p', type=float, default=0.5,
                          help='Dropout probability for fully connected layers')
        
        # General model params
        group.add_argument('--activation', type=str, default='LeakyReLU',
                          help='Activation function')
        group.add_argument('--use_batch_norm', type=utils.str2bool, default=True,
                          help='Use batch normalization')
        group.add_argument('--use_weight_norm', type=utils.str2bool, default=False,
                          help='Use weight normalization')
        
        # Loss function params
        group.add_argument('--loss_criterion', type=str, default='MSELoss',
                          help='Loss criterion')
        
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        """
        Add conditional arguments based on known arguments.
        
        Args:
            parser (argparse.ArgumentParser): The argument parser.
            known_args (argparse.Namespace): Known arguments.
            
        Returns:
            argparse.ArgumentParser: The argument parser with added conditional arguments.
        """
        parser = add_criterion_specific_args(parser, known_args.loss_criterion)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process grouped arguments and extract model-specific arguments.
        
        Args:
            grouped_args (dict): Dictionary of grouped arguments.
            
        Returns:
            argparse.Namespace: Namespace containing model-specific arguments.
        """
        model_args = grouped_args['Model Module args']
        model_args.loss_args = vars(grouped_args['Criterion args'])
        return model_args

    def __init__(self, 
                 input_len=84,
                 n_res_blocks=3,
                 filters_per_block=[64, 128, 256],
                 kernel_sizes=[3, 3, 3],
                 stride_sizes=[1, 1, 1],
                 fc_layers=[128, 64],
                 n_outputs=1,
                 conv_dropout_p=0.1,
                 fc_dropout_p=0.5,
                 activation='LeakyReLU',
                 use_batch_norm=True,
                 use_weight_norm=False,
                 loss_criterion='MSELoss',
                 loss_args={}):
        """
        Initialize the ResNetModule.
        
        Args:
            input_len (int): Fixed sequence length of inputs.
            n_res_blocks (int): Number of residual blocks.
            filters_per_block (list): Number of filters for each residual block.
            kernel_sizes (list): Kernel sizes for each residual block.
            stride_sizes (list): Stride sizes for each residual block.
            fc_layers (list): Number of units in fully connected layers.
            n_outputs (int): Number of output units.
            conv_dropout_p (float): Dropout probability for convolutional layers.
            fc_dropout_p (float): Dropout probability for fully connected layers.
            activation (str): Activation function name.
            use_batch_norm (bool): Whether to use batch normalization.
            use_weight_norm (bool): Whether to use weight normalization.
            loss_criterion (str): Loss criterion name.
            loss_args (dict): Arguments for the loss criterion.
        """
        super().__init__()
        
        # Save initialization parameters
        self.input_len = input_len
        self.n_res_blocks = n_res_blocks
        self.filters_per_block = filters_per_block
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.fc_layers = fc_layers
        self.n_outputs = n_outputs
        self.conv_dropout_p = conv_dropout_p
        self.fc_dropout_p = fc_dropout_p
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm
        self.loss_criterion = loss_criterion
        self.loss_args = loss_args
        
        # Ensure parameter lists have correct lengths
        if len(self.filters_per_block) != self.n_res_blocks:
            raise ValueError(f"filters_per_block must have length n_res_blocks ({self.n_res_blocks})")
        if len(self.kernel_sizes) != self.n_res_blocks:
            raise ValueError(f"kernel_sizes must have length n_res_blocks ({self.n_res_blocks})")
        if len(self.stride_sizes) != self.n_res_blocks:
            self.stride_sizes = [1] * self.n_res_blocks
        
        # Build the residual blocks
        res_blocks = []
        in_channels = 4  # 4 for one-hot encoded DNA
        
        for i in range(self.n_res_blocks):
            out_channels = self.filters_per_block[i]
            kernel_size = self.kernel_sizes[i]
            stride = self.stride_sizes[i]
            
            res_blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dropout_p=self.conv_dropout_p,
                    activation=self.activation,
                    use_batch_norm=self.use_batch_norm,
                    use_weight_norm=self.use_weight_norm
                )
            )
            
            in_channels = out_channels
        
        self.res_stack = nn.Sequential(*res_blocks)
        
        # Calculate output feature dimension after residual blocks
        # Start with input_len and apply stride reductions
        feature_dim = self.input_len
        for stride in self.stride_sizes:
            feature_dim = math.ceil(feature_dim / stride)
        
        # Multiply by number of output channels from last residual block
        flattened_dim = feature_dim * self.filters_per_block[-1]
        
        # Build fully connected layers
        fc_layers_list = []
        in_dim = flattened_dim
        
        for out_dim in self.fc_layers:
            fc_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim) if self.use_batch_norm else nn.Identity(),
                getattr(nn, self.activation)(),
                nn.Dropout(self.fc_dropout_p)
            ])
            in_dim = out_dim
        
        # Add final output layer
        fc_layers_list.append(nn.Linear(in_dim, self.n_outputs))
        
        self.fc_stack = nn.Sequential(*fc_layers_list)
        
        # Set up loss function
        self.criterion = getattr(loss_functions, self.loss_criterion)(**self.loss_args)
    
    def forward(self, x):
        """
        Forward pass through the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 4, sequence_length].
            
        Returns:
            torch.Tensor: Output tensor.
        """
        # Ensure input is in the right format: [batch_size, channels, sequence_length]
        if x.shape[1] == self.input_len and x.shape[2] == 4:
            # Input is [batch_size, length, channels], permute to [batch_size, channels, length]
            x = x.permute(0, 2, 1)
        
        # Process through residual blocks
        x = self.res_stack(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Process through fully connected layers
        x = self.fc_stack(x)
        
        return x