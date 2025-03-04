#!/usr/bin/env python3
"""
Enhanced training script with Weights & Biases integration for BODA2.
Supports hyperparameter sweeps across different data modules, model architectures,
and training configurations.
"""
import os
import sys
import re
import time
import shutil
import argparse
import tarfile
import tempfile
import random
import subprocess
import ast
from typing import Dict, Any, List, Union, Optional

import torch
from lightning.pytorch import Trainer
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import boda
from boda.common import utils
from boda.common.utils import unpack_artifact, model_fn

import hypertune
import wandb
from lightning.pytorch.loggers import WandbLogger

#####################
# Helper Functions  #
#####################

def convert_to_list(param):
    """
    Convert a parameter into a list.
    
    This function handles various input formats:
    - Lists of strings that might be representations of lists
    - Space-separated strings
    - Actual list objects
    
    Args:
        param: The parameter to convert (string, list, or other)
        
    Returns:
        list: The converted parameter as a list
    """
    if isinstance(param, list):
        flattened = []
        for item in param:
            if isinstance(item, str):
                item = item.strip()
                if item.startswith('[') and item.endswith(']'):
                    try:
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, list):
                            flattened.extend(parsed)
                        else:
                            flattened.append(item)
                    except Exception:
                        flattened.append(item)
                else:
                    flattened.append(item)
            else:
                flattened.append(item)
        return flattened
    elif isinstance(param, str):
        param = param.strip()
        if param.startswith('[') and param.endswith(']'):
            try:
                return ast.literal_eval(param)
            except Exception:
                return param.split()
        else:
            return param.split()
    else:
        return param

def set_best(my_model, callbacks):
    """
    Set the model to the best checkpoint based on the monitored metric.
    
    Args:
        my_model: The model to update
        callbacks: Dictionary of callbacks including 'model_checkpoint'
        
    Returns:
        The updated model with weights from the best checkpoint
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = callbacks['model_checkpoint'].best_model_path
            get_epoch = re.search('epoch=(\d*)', best_path).group(1)
            if 'gs://' in best_path:
                subprocess.call(['gsutil','cp',best_path,tmpdirname])
                best_path = os.path.join(tmpdirname, os.path.basename(best_path))
            print(f'Best model stashed at: {best_path}', file=sys.stderr)
            print(f'Exists: {os.path.isfile(best_path)}', file=sys.stderr)
            ckpt = torch.load(best_path)
            my_model.load_state_dict(ckpt['state_dict'])
            print(f'Setting model from epoch: {get_epoch}', file=sys.stderr)
        except KeyError:
            print('Setting most recent model', file=sys.stderr)
    return my_model

def save_model(data_module, model_module, graph_module, model, trainer, args):
    """
    Save the model and associated artifacts.
    
    This function creates a checkpoint file with:
    - Model state dictionary
    - Data module information and hyperparameters
    - Model module information and hyperparameters
    - Graph module information and hyperparameters
    - Timestamp and random tag
    
    It then compresses these into a tar.gz file and saves it to the specified path.
    
    Args:
        data_module: The data module class
        model_module: The model module class
        graph_module: The graph module class
        model: The trained model
        trainer: The PyTorch Lightning trainer
        args: Dictionary of input arguments
    """
    # Get the root directory from Trainer args
    local_dir = args['pl.Trainer'].default_root_dir
    
    # Create a dictionary with all relevant information
    save_dict = {
        'data_module': data_module.__name__,
        'data_hparams': data_module.process_args(args),
        'model_module': model_module.__name__,
        'model_hparams': model_module.process_args(args),
        'graph_module': graph_module.__name__,
        'graph_hparams': graph_module.process_args(args),
        'model_state_dict': model.state_dict(),
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'random_tag': random.randint(100000, 999999)
    }
    
    # Save the checkpoint
    torch.save(save_dict, os.path.join(local_dir, 'torch_checkpoint.pt'))
    
    # Create a compressed archive
    filename = f'model_artifacts__{save_dict["timestamp"]}__{save_dict["random_tag"]}.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(os.path.join(tmpdirname, filename), 'w:gz') as tar:
            tar.add(local_dir, arcname='artifacts')

        # Copy to destination (either Google Cloud Storage or local path)
        if 'gs://' in args['Main args'].artifact_path:
            cloud_target = os.path.join(args['Main args'].artifact_path, filename)
            subprocess.check_call(['gsutil', 'cp', os.path.join(tmpdirname, filename), cloud_target])
        else:
            os.makedirs(args['Main args'].artifact_path, exist_ok=True)
            shutil.copy(os.path.join(tmpdirname, filename), args['Main args'].artifact_path)
    
    print(f"Model saved to {args['Main args'].artifact_path}/{filename}")

#######################
# Main Training Logic #
#######################

def main(args):
    """
    Main function for training a model using Pytorch Lightning with W&B integration.
    
    Args:
        args: Dictionary containing all input arguments organized by module
        
    Returns:
        None
    """
    # Get the module classes
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module = getattr(boda.model, args['Main args'].model_module)
    graph_module = getattr(boda.graph, args['Main args'].graph_module)

    # Initialize data module
    data_args = data_module.process_args(args)
    
    # Special handling for list-type arguments that might come from YAML config
    if hasattr(data_args, 'activity_columns'):
        data_args.activity_columns = convert_to_list(data_args.activity_columns)
    if hasattr(data_args, 'stderr_columns'):
        data_args.stderr_columns = convert_to_list(data_args.stderr_columns)
    if hasattr(data_args, 'val_chrs'):
        data_args.val_chrs = convert_to_list(data_args.val_chrs)
    if hasattr(data_args, 'test_chrs'):
        data_args.test_chrs = convert_to_list(data_args.test_chrs)
    
    data = data_module(**vars(data_args))
    
    # Initialize model
    model = model_module(**vars(model_module.process_args(args)))
    
    # Initialize graph module with model
    graph = graph_module(model=model, **vars(graph_module.process_args(args)))

    # Set up logger based on command-line input
    if args['Main args'].logger_type.lower() == 'wandb':
        # Generate a unique run ID and process the run name
        run_id = wandb.util.generate_id()
        run_name = args['Main args'].run_name.replace("{runid}", run_id)
        
        logger = WandbLogger(
            project=args['Main args'].logger_project,
            name=run_name,  # Now using the processed run_name
            log_model=True
        )
        
        # Log all hyperparameters to W&B
        logger.log_hyperparams(vars(args['Main args']))
        if 'Model Module args' in args:
            logger.log_hyperparams(vars(args['Model Module args']))
        if 'Data Module args' in args or args['Main args'].data_module == 'PromoterDataModule':
            module_name = 'Data Module args'
            if args['Main args'].data_module == 'PromoterDataModule':
                module_name = 'Promoter DataModule'
            if module_name in args:
                logger.log_hyperparams(vars(args[module_name]))
        if 'Graph Module args' in args:
            logger.log_hyperparams(vars(args['Graph Module args']))
    elif args['Main args'].logger_type.lower() == 'tensorboard':
        logger = pl_loggers.TensorBoardLogger(
            save_dir='./logs',
            name=args['Main args'].logger_project
        )
    else:
        logger = True  # Default Lightning logger

    print(f"Original run_name: {args['Main args'].run_name}")
    print(f"Generated run_id: {run_id}")
    print(f"Processed run_name: {run_name}")

    # Set up callbacks
    use_callbacks = {
        'learning_rate_monitor': LearningRateMonitor()
    }
    
    if args['Main args'].checkpoint_monitor is not None:
        use_callbacks['model_checkpoint'] = ModelCheckpoint(
            save_top_k=1,
            monitor=args['Main args'].checkpoint_monitor,
            mode=args['Main args'].stopping_mode
        )
        use_callbacks['early_stopping'] = EarlyStopping(
            monitor=args['Main args'].checkpoint_monitor,
            patience=args['Main args'].stopping_patience,
            mode=args['Main args'].stopping_mode
        )

    # Ensure output directory exists
    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    
    # Create trainer
    trainer = Trainer.from_argparse_args(
        args['pl.Trainer'],
        callbacks=list(use_callbacks.values()),
        logger=logger
    )

    # Train the model
    trainer.fit(graph, data)
    
    # Load the best model
    graph = set_best(graph, use_callbacks)

    # Report metrics and save the model
    try:
        if 'model_checkpoint' in use_callbacks:
            mc_dict = vars(use_callbacks['model_checkpoint'])
            keys = ['monitor', 'best_model_score']
            tag, metric = [mc_dict[key] for key in keys]
            
            # Report to hypertune if available
            try:
                graph.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=tag,
                    metric_value=metric.item(),
                    global_step=graph.global_step + 1
                )
            except (AttributeError, NameError):
                pass
                
            print(f'{tag} at {graph.global_step}: {metric}', file=sys.stderr)
    except (KeyError, AttributeError):
        print("Couldn't report best metric, using final model state", file=sys.stderr)

    # Save the model
    save_model(data_module, model_module, graph_module, graph.model, trainer, args)
    
    # Finish W&B run if active
    if args['Main args'].logger_type.lower() == 'wandb' and wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    # Build the base parser
    parser = argparse.ArgumentParser(description="BODA trainer with W&B integration", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True,
                       help='BODA data module to process dataset.')
    group.add_argument('--model_module', type=str, required=True,
                       help='BODA model module to fit dataset.')
    group.add_argument('--graph_module', type=str, required=True,
                       help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/',
                       help='Path where model artifacts are deposited.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    group.add_argument('--checkpoint_monitor', type=str,
                       help='String to monitor PTL logs if saving best.')
    group.add_argument('--stopping_mode', type=str, default='min',
                       help='Goal for monitored metric e.g. (max or min).')
    group.add_argument('--stopping_patience', type=int, default=100,
                       help='Number of epochs of non-improvement tolerated before early stopping.')
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False,
                       help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')

    # New logger control arguments (renamed to avoid conflict with Trainer's logger)
    group.add_argument('--logger_type', type=str, default='wandb',
                       help='Which logger to use (wandb, tensorboard, none)')
    group.add_argument('--logger_project', type=str, default='boda_train',
                       help='Project name for the logger.')
    group.add_argument('--run_name', type=str, default='default_run',
                       help='Run name for the logger.')

    # Parse initial arguments to get module classes
    known_args, leftover_args = parser.parse_known_args()
    
    # Import the respective modules
    try:
        Data = getattr(boda.data, known_args.data_module)
        Model = getattr(boda.model, known_args.model_module)
        Graph = getattr(boda.graph, known_args.graph_module)
    except AttributeError as e:
        print(f"Error: {str(e)}")
        print(f"Available data modules: {dir(boda.data)}")
        print(f"Available model modules: {dir(boda.model)}")
        print(f"Available graph modules: {dir(boda.graph)}")
        sys.exit(1)

    # Add module-specific arguments
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    # Get updated known arguments
    known_args, leftover_args = parser.parse_known_args()
    
    # Add conditional arguments based on the known arguments
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    # Add Trainer-specific arguments
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    
    # Parse all arguments
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
    
    # Organize arguments into groups
    args = utils.organize_args(parser, args)
    
    # Print argument summary
    print('-' * 80)
    print("Starting training with configuration:")
    for group_title, namespace_obj in args.items():
        print(f"\n{group_title}:")
        for key, value in sorted(vars(namespace_obj).items()):
            print(f"  {key}: {value}")
    print('-' * 80)
    
    # Start the training
    main(args)
