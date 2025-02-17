#!/usr/bin/env python3
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
# PTL Module saving #
#####################

def set_best(my_model, callbacks):
    """
    Set the best model checkpoint for the provided model.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = callbacks['model_checkpoint'].best_model_path
            get_epoch = re.search('epoch=(\d*)', best_path).group(1)
            if 'gs://' in best_path:
                subprocess.call(['gsutil', 'cp', best_path, tmpdirname])
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
    """
    # Retrieve the default root directory from Trainer args.
    local_dir = args['pl.Trainer'].default_root_dir
    # Build a dictionary containing module information, hyperparameters, and model state.
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
    torch.save(save_dict, os.path.join(local_dir, 'torch_checkpoint.pt'))
    
    filename = f'model_artifacts__{save_dict["timestamp"]}__{save_dict["random_tag"]}.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(os.path.join(tmpdirname, filename), 'w:gz') as tar:
            tar.add(local_dir, arcname='artifacts')
        if 'gs://' in args['Main args'].artifact_path:
            cloud_target = os.path.join(args['Main args'].artifact_path, filename)
            subprocess.check_call(['gsutil', 'cp', os.path.join(tmpdirname, filename), cloud_target])
        else:
            os.makedirs(args['Main args'].artifact_path, exist_ok=True)
            shutil.copy(os.path.join(tmpdirname, filename), args['Main args'].artifact_path)

#################################
# Argument processing for lists #
#################################

def convert_to_list(param):
    """
    Convert a parameter into a list.
    
    If param is a list, iterate over its elements: if any element is a string
    that starts with '[' and ends with ']', evaluate it with ast.literal_eval
    and flatten the result. Otherwise, if param is a string, split it on whitespace,
    or try to ast.literal_eval it if it appears to be a list.
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
    
#######################
# Main and run blocks #
#######################

def main(args):
    # Load modules from the BODA library using the specified names.
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module = getattr(boda.model, args['Main args'].model_module)
    graph_module = getattr(boda.graph, args['Main args'].graph_module)

    # Initialize the modules using their processed arguments.
    data = data_module(**vars(data_module.process_args(args)))
    model = model_module(**vars(model_module.process_args(args)))
    graph = graph_module(model=model, **vars(graph_module.process_args(args)))

    # Set up the logger based on command-line input.
    if args['Main args'].logger_type.lower() == 'wandb':
        logger = WandbLogger(
            project=args['Main args'].logger_project,
            name=args['Main args'].run_name,
            log_model=True
        )
    elif args['Main args'].logger_type.lower() == 'tensorboard':
        logger = pl_loggers.TensorBoardLogger(
            save_dir='./logs',
            name=args['Main args'].logger_project
        )
    else:
        logger = None

    # Set up callbacks.
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

    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    # Create the Trainer from its specific subset of arguments.
    trainer = Trainer.from_argparse_args(
        args['pl.Trainer'],
        callbacks=list(use_callbacks.values()),
        logger=logger
    )

    trainer.fit(graph, data)
    graph = set_best(graph, use_callbacks)

    try:
        mc_dict = vars(use_callbacks['model_checkpoint'])
        keys = ['monitor', 'best_model_score']
        tag, metric = [mc_dict[key] for key in keys]
        graph.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=tag,
            metric_value=metric.item(),
            global_step=graph.global_step + 1)
        print(f'{tag} at {graph.global_step}: {metric}', file=sys.stderr)
    except KeyError:
        print('Used default checkpointing.', file=sys.stderr)
    except AttributeError:
        print("No hypertune instance found.", file=sys.stderr)

    save_model(data_module, model_module, graph_module, graph.model, trainer, args)

if __name__ == '__main__':
    # Build the base parser.
    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
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

    # (Module-specific hyperparameters are added by the module-specific functions.)
    
    # Retrieve the module classes based on initial known arguments.
    known_args, leftover_args = parser.parse_known_args()
    Data = getattr(boda.data, known_args.data_module)
    Model = getattr(boda.model, known_args.model_module)
    Graph = getattr(boda.graph, known_args.graph_module)

    # Add module-specific arguments.
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    known_args, leftover_args = parser.parse_known_args()
    
    # Add conditional arguments based on the known arguments.
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    # Append Trainer-specific arguments.
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
    
    # Organize arguments into groups (e.g., 'pl.Trainer') using your utility.
    args = utils.organize_args(parser, args)
    
    print('-' * 80)
    print("Organized arguments: \n")
    # (Optional) inspect the organized groups:
    for group_title, namespace_obj in args.items():
        print(group_title, vars(namespace_obj))
        print("")
    print('-' * 80)
    
    # Since "activity_columns", "stderr_columns", "val_chrs", and "test_chrs" are in "Data Module args":
    data_args = args["Data Module args"]
    data_args.activity_columns = convert_to_list(data_args.activity_columns)
    data_args.stderr_columns   = convert_to_list(data_args.stderr_columns)
    data_args.val_chrs         = convert_to_list(data_args.val_chrs)
    data_args.test_chrs        = convert_to_list(data_args.test_chrs)
    print('')

    main(args)
    if wandb.run is not None:
        wandb.finish()
