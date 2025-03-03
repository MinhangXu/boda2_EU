import os
import sys
import re
import time
import yaml
import shutil
import argparse
import tarfile
import tempfile
import random
import subprocess

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

    This function sets the state of the provided model to the state of the best checkpoint,
    as determined by the `ModelCheckpoint` callback.

    Args:
        my_model (nn.Module): The model to be updated.
        callbacks (dict): Dictionary of callbacks, including 'model_checkpoint'.

    Returns:
        nn.Module: The updated model.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = callbacks['model_checkpoint'].best_model_path
            get_epoch = re.search('epoch=(\d*)', best_path).group(1)
            if 'gs://' in best_path:
                subprocess.call(['gsutil','cp',best_path,tmpdirname])
                best_path = os.path.join( tmpdirname, os.path.basename(best_path) )
            print(f'Best model stashed at: {best_path}', file=sys.stderr)
            print(f'Exists: {os.path.isfile(best_path)}', file=sys.stderr)
            ckpt = torch.load( best_path )
            my_model.load_state_dict( ckpt['state_dict'] )
            print(f'Setting model from epoch: {get_epoch}', file=sys.stderr)
        except KeyError:
            print('Setting most recent model', file=sys.stderr)
    return my_model

def save_model(data_module, model_module, graph_module, 
                model, trainer, args):
    """
    Save the model and associated artifacts.

    This function saves the model's state dictionary, along with information about the data, model, and graph modules,
    to a checkpoint file. Additionally, it compresses the artifacts directory and saves it as a tar.gz file.

    Args:
        data_module (Module): Data module class.
        model_module (Module): Model module class.
        graph_module (Module): Graph module class.
        model (nn.Module): The model to be saved.
        trainer (lightning.pytorch.Trainer): The Trainer instance used for training.
        args (dict): Dictionary of input arguments.

    Returns:
        None
    """
    local_dir = args['pl.Trainer'].default_root_dir
    # 1) Build a dictionary with model_state_dict, plus references to data & model modules
    save_dict = {
        'data_module'  : data_module.__name__,
        'data_hparams' : data_module.process_args(args),
        'model_module' : model_module.__name__,
        'model_hparams': model_module.process_args(args),
        'graph_module' : graph_module.__name__,
        'graph_hparams': graph_module.process_args(args),
        'model_state_dict': model.state_dict(),
        'timestamp'    : time.strftime("%Y%m%d_%H%M%S"),
        'random_tag'   : random.randint(100000,999999)
    }
    torch.save(save_dict, os.path.join(local_dir,'torch_checkpoint.pt'))
    
    # 2) Tar it up into model_artifacts__TIMESTAMP__RANDOM.tar.gz
    filename=f'model_artifacts__{save_dict["timestamp"]}__{save_dict["random_tag"]}_nov11.tar.gz'
    with tempfile.TemporaryDirectory() as tmpdirname:
        with tarfile.open(os.path.join(tmpdirname,filename), 'w:gz') as tar:
            tar.add(local_dir,arcname='artifacts')

        if 'gs://' in args['Main args'].artifact_path:
            clound_target = os.path.join(args['Main args'].artifact_path,filename)
            subprocess.check_call(
                ['gsutil', 'cp', os.path.join(tmpdirname,filename), clound_target]
            )
        else:
            # save to local artifact path that's defined in the CLI. 
            os.makedirs(args['Main args'].artifact_path, exist_ok=True)
            shutil.copy(os.path.join(tmpdirname,filename), args['Main args'].artifact_path)

#######################
# Main and run blocks #
# ######################

def main(args):
    """
    Main function for training a model using Pytorch Lightning.

    This function orchestrates the training of a model using the specified data, model, and graph modules.
    It sets up callbacks, creates the Trainer instance, fits the model, reports hypertuning metrics if applicable,
    and saves the trained model and artifacts.

    Args:
        args (dict): Dictionary of input arguments.

    Returns:
        None
    """
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module= getattr(boda.model, args['Main args'].model_module)
    graph_module= getattr(boda.graph, args['Main args'].graph_module)

    data = data_module(**vars(data_module.process_args(args)))
    #print('here')
    model= model_module(**vars(model_module.process_args(args)))
    #print("after model")
    graph = graph_module(
        model = model,
        **vars(graph_module.process_args(args))
    )
    #print("after graph")

    wandb_logger = WandbLogger(
        project='boda_train',
        name='test_performance_nov11', 
        log_model=True   
    )
    
    # 
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
    
    # tb_logger = True
        
    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    trainer = Trainer.from_argparse_args(
        args['pl.Trainer'], 
        callbacks=list(use_callbacks.values()),
        logger=wandb_logger     # use wandb logger
    )
    #print('before fit')
    trainer.fit(graph, data)
    
    graph = set_best(graph, use_callbacks)

    # Report hyperparameter tuning metric
    try:
        mc_dict = vars(use_callbacks['model_checkpoint'])
        keys = ['monitor', 'best_model_score']
        tag, metric = [ mc_dict[key] for key in keys ]
        graph.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=tag,
            metric_value=metric.item(),
            global_step=graph.global_step + 1)
        print(f'{tag} at {graph.global_step}: {metric}', file=sys.stderr)
    except KeyError:
        print('Used default checkpointing.', file=sys.stderr)
    except AttributeError:
        print("No hypertune instance found.", file=sys.stderr)
        pass
    
    save_model(data_module, model_module, graph_module, 
               graph.model, trainer, args)
    
if __name__ == '__main__':
    wandb.init()

    parser = argparse.ArgumentParser(description="BODA trainer", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True, help='BODA data module to process dataset.')
    group.add_argument('--model_module',type=str, required=True, help='BODA model module to fit dataset.')
    group.add_argument('--graph_module',type=str, required=True, help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/', help='Path where model artifacts are deposited.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    group.add_argument('--checkpoint_monitor', type=str, help='String to monior PTL logs if saving best.')
    group.add_argument('--stopping_mode', type=str, default='min', help='Goal for monitored metric e.g. (max or min).')
    group.add_argument('--stopping_patience', type=int, default=100, help='Number of epochs of non-improvement tolerated before early stopping.')
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False, help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')
    known_args, leftover_args = parser.parse_known_args()
    
    Data  = getattr(boda.data,  known_args.data_module)
    Model = getattr(boda.model, known_args.model_module)
    Graph = getattr(boda.graph, known_args.graph_module)
    
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    known_args, leftover_args = parser.parse_known_args()
    
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    
    if known_args.tolerate_unknown_args:
        args, leftover_args = parser.parse_known_args()
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        args = parser.parse_args()
    
    print("before ", type(args))
    args = boda.common.utils.organize_args(parser, args)
    print("after ", type(args))
    main(args)

    wandb.finish()