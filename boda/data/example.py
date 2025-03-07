r"""Example of a standardized module to process data for training models. 
This is based on the spec from lightning.pytorch.core.datamodule (see 
https://lightning.ai/docs for more information) with additional staticmethods 
that can add arguments to an ArgumentParser which mirror class construction 
arguments.
"""

import torch
import argparse

class ExampleData(torch.nn.Module):
    @staticmethod
    def add_data_specific_args(parent_parser):
        r"""Argparse interface to expose class kwargs to the command line.
        Required to specify for compatability with CLI. These args should 
        mirror class construction arguments that are not determined 
        conditionally.
        Args:
            parent_parser (argparse.ArgumentParser): A parent ArgumentParser to 
                which more args will be added.
        Returns:
            An updated ArgumentParser
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Data Module args')
        group.add_argument('--data_dir', type=str)
        group.add_argument('--batch_size', type=int)
        return parser
        
    @staticmethod
    def add_conditional_args(parser, known_args):
        r"""Method to add conditional args after `add_data_specific args` 
        sets primary arguments. Required to specify for CLI conpatability, 
        but can return unmodified parser as in this example. These args should 
        mirror class construction arguments that would be determined based on 
        other arguments.
        Args:
            parser (argparse.AtgumentParser): an ArgumentParser to  which 
                more args can be added.
            known_args (Namespace): Known arguments that have been parsed 
                by non-conditionally specified args.
        """
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Perform any required processessing of command line args required 
        before passing to the class constructor. A bridge between the parsed 
        arguments and the class constructor.

        Args:
            grouped_args (Namespace): Namespace of known arguments with 
            `'Data Module args'` key.

        Returns:
            Namespace: A modified namespace that can be passed to the 
            associated class constructor.
        """
        model_args   = grouped_args['Data Module args']
        return model_args

    def __init__(self, data_dir='/no/data_dir/specified', batch_size=8, **kwargs):
        super().__init__()
        delf.data_name = 'ExampleData'
        self.data_dir  = data_dir
        self.batch_size= batch_size
        
        self.unused_kwargs = kwargs

    def prepare_data(self):
        r"""A method to prepare data based on class attributes, usually by 
        linking new attributes to `self`.
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
        
    def train_dataloader(self):
        r"""Method that returns a DataLoader for training data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            Training_DataLoader
        """
        raise NotImplementedError
        return Training_DataLoader
        
    def val_dataloader(self):
        r"""Method that returns a DataLoader for validation data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            Validation_DataLoader
        """
        raise NotImplementedError
        return Validation_DataLoader
        
    def test_dataloader(self):
        r"""Method that returns a DataLoader for testing data split 
        (e.g., a properly formatted torch.utils.data.DataLoader)
        Args:
            None
        Returns:
            Testing_DataLoader
        """
        raise NotImplementedError
        return Testing_DataLoader
