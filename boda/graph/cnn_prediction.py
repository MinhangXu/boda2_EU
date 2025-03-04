import argparse
import math
import os
import sys
import time
import tempfile
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch import LightningModule

import hypertune

from ..common import utils
from .utils import (add_optimizer_specific_args, add_scheduler_specific_args, reorg_optimizer_args, reorg_scheduler_args,
                    filter_state_dict, pearson_correlation, spearman_correlation, shannon_entropy, r2_score)

class CNNBasicTraining(LightningModule):
    """
    LightningModule for basic training of a CNN model.

    Args:
        optimizer (str): Name of the optimizer. Default is 'Adam'.
        scheduler (str): Name of the learning rate scheduler. Default is None.
        scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
        scheduler_interval (str): Scheduler interval. Default is 'epoch'.
        optimizer_args (dict): Arguments for the optimizer. Default is None.
        scheduler_args (dict): Arguments for the scheduler. Default is None.
    """
    
    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        """
        Add command-line arguments specific to the Graph module.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')
        return parser
    
    @staticmethod
    def add_conditional_args(parser, known_args):
        """
        Add conditional arguments to the parser based on known arguments.

        Args:
            parser (argparse.ArgumentParser): Argument parser.
            known_args (Namespace): Known arguments.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = add_optimizer_specific_args(parser, known_args.optimizer)
        parser = add_scheduler_specific_args(parser, known_args.scheduler)
        return parser

    @staticmethod
    def process_args(grouped_args):
        """
        Process the command-line arguments for the Graph module.

        Args:
            grouped_args (dict): Grouped command-line arguments.

        Returns:
            Namespace: Processed arguments.
        """
        graph_args   = grouped_args['Graph Module args']
        graph_args.optimizer_args = vars(grouped_args['Optimizer args'])
        graph_args.optimizer_args = reorg_optimizer_args(graph_args.optimizer_args)
        try:
            graph_args.scheduler_args = vars(grouped_args['LR Scheduler args'])
            graph_args.scheduler_args = reorg_scheduler_args(graph_args.scheduler_args)
        except KeyError:
            graph_args.scheduler_args = None
        return graph_args

    ####################
    # Standard methods #
    ####################
    
    def __init__(self, model, optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        """
        Initialize the CNNBasicTraining module.

        Args:
            model (torch.nn.Module): A torch or lightning.pytorch Module.
            optimizer (str): Name of the optimizer. Default is 'Adam'.
            scheduler (str): Name of the learning rate scheduler. Default is None.
            scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
            scheduler_interval (str): Scheduler interval. Default is 'epoch'.
            optimizer_args (dict): Arguments for the optimizer. Default is None.
            scheduler_args (dict): Arguments for the scheduler. Default is None.
        """
        super().__init__()
        self.model = model
        self.criterion = model.criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval= scheduler_interval
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        
    def forward(self, input):
        """
        Set forward call.
        
        Args:
            input (tensor): Input tensor for model.
        """
        return self.model(input)

    ###################
    # Non-PTL methods #
    ###################
        
    def categorical_mse(self, x, y):
        """
        Calculate the categorical mean squared error between x and y.

        Args:
            x (torch.Tensor): Input tensor x.
            y (torch.Tensor): Input tensor y.

        Returns:
            torch.Tensor: Categorical mean squared error.
        """
        return (x - y).pow(2).mean(dim=0)
        
    def aug_log(self, internal_metrics=None, external_metrics=None):
        """
        Log metrics for hyperparameter tuning and printing.

        Args:
            internal_metrics (dict, optional): Internal metrics dictionary. Default is None.
            external_metrics (dict, optional): External metrics dictionary. Default is None.
        """
        if internal_metrics is not None:
            for my_key, my_value in internal_metrics.items():
                self.log(my_key, my_value)
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=my_key,
                    metric_value=my_value,
                    global_step=self.global_step)
                
        if external_metrics is not None:
            res_str = '|'
            for my_key, my_value in external_metrics.items():
                self.log(my_key, my_value)
                res_str += ' {}: {:.5f} |'.format(my_key, my_value)
                self.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=my_key,
                    metric_value=my_value,
                    global_step=self.global_step)
            border = '-'*len(res_str)
            print("\n".join(['',border, res_str, border,'']))
        
        return None

    #############
    # PTL hooks #
    #############
        
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Union[Optimizer, Tuple[List[Optimizer], List[Dict]]]: Optimizer(s) and scheduler(s).
        """
        self.hpt = hypertune.HyperTune()
        params = [ x for x in self.parameters() if x.requires_grad ]
        print(f'Found {sum(p.numel() for p in params)} parameters')
        optim_class = getattr(torch.optim,self.optimizer)
        my_optimizer= optim_class(self.parameters(), **self.optimizer_args)
        if self.scheduler is not None:
            sch_dict = {
                'scheduler': getattr(torch.optim.lr_scheduler,self.scheduler)(my_optimizer, **self.scheduler_args), 
                'interval': self.scheduler_interval, 
                'name': 'learning_rate'
            }
            if self.scheduler_monitor is not None:
                sch_dict['monitor'] = self.scheduler_monitor
            return [my_optimizer], [sch_dict]
        else:
            return my_optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Training step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss for the training step.
        """
        x, y   = batch
        y_hat  = self(x)
        loss   = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Validation step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.

        Returns:
            dict: Dictionary containing loss, metric, predictions, and labels for the validation step.
        """
        x, y   = batch
        y_hat = self(x)

        # If y_hat is 2D and y is 1D, squeeze y_hat
        if y_hat.dim() == 2 and y_hat.shape[1] == 1 and y.dim() == 1:
            y_hat = y_hat.squeeze(1)

        # Loss
        loss   = self.criterion(y_hat, y)
        self.log('step_valid_loss', loss)

        # R2 score
        r2 = r2_score(y_hat, y)
        self.log('step_valid_r2', r2)

        # Pearson correlation
        pearsonr_vals, mean_pearsonr = pearson_correlation(y_hat, y) 
        self.log('valid_mean_pearson', mean_pearsonr)

        # per cell-type pearson correlation
        n_outputs = getattr(self.model, 'n_outputs', 1)  # or store in hparams
        if n_outputs == 3:
            cell_types = ['K562', 'HepG2', 'SKNSH']
            for i, coeff in enumerate(pearsonr_vals):
                # If i >= len(cell_types), you'll get an index error
                # so be sure you have exactly i elements in cell_types.
                self.log(f'valid_pearson_{cell_types[i]}', coeff)
                self.log(f'valid_pearson_squared_{cell_types[i]}', coeff**2)
        else:
            # Single output scenario
            cell_types = ['SingleOutput']

        metric = self.categorical_mse(y_hat, y)
        return {'loss': loss, 'metric': metric, 'preds': y_hat, 'labels': y}

    def validation_epoch_end(self, val_step_outputs):
        """
        Called at the end of the validation epoch.

        Args:
            val_step_outputs (list): List of dictionaries containing validation step outputs.
        """
        arit_mean = torch.stack([ batch['loss'] for batch in val_step_outputs ], dim=0) \
                      .mean()
        harm_mean = torch.stack([ batch['metric'] for batch in val_step_outputs ], dim=0) \
                      .mean(dim=0).pow(-1).mean().pow(-1)
        epoch_preds = torch.cat([batch['preds'] for batch in val_step_outputs], dim=0)
        epoch_labels  = torch.cat([batch['labels'] for batch in val_step_outputs], dim=0)

        # Compute R² score
        r2_val_score = r2_score(epoch_labels, epoch_preds)

        spearman, mean_spearman = spearman_correlation(epoch_preds, epoch_labels)
        shannon_pred, shannon_label = shannon_entropy(epoch_preds), shannon_entropy(epoch_labels)
        specificity_spearman, specificity_mean_spearman = spearman_correlation(shannon_pred, shannon_label)
        '''
        self.aug_log(external_metrics={
            'current_epoch': self.current_epoch, 
            'arithmetic_mean_loss': arit_mean,
            'harmonic_mean_loss': harm_mean,
            'prediction_mean_spearman': mean_spearman.item(),
            'entropy_spearman': specificity_mean_spearman.item(), 
            'epoch_end_r2': r2_val_score
        })
        '''
        # Log with epoch as x-axis
        on_epoch = True  # This ensures metrics are logged per epoch
        self.log('epoch_end_r2', r2_val_score, on_epoch=on_epoch)
        self.log('arithmetic_mean_loss', arit_mean, on_epoch=on_epoch)
        self.log('harmonic_mean_loss', harm_mean, on_epoch=on_epoch) 
        self.log('prediction_mean_spearman', mean_spearman.item(), on_epoch=on_epoch)
        self.log('entropy_spearman', specificity_mean_spearman.item(), on_epoch=on_epoch)
        
        # You can also explicitly log the epoch
        self.log('current_epoch', self.current_epoch, on_epoch=True)

        return None
    
    def test_step(self, batch, batch_idx):
        """
        Test step implementation.

        Args:
            batch: Batch of data.
            batch_idx (int): Batch index.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log('test_loss', loss)       

class CNNTransferLearning(CNNBasicTraining):
    """
    LightningModule for transfer learning with a CNN model.

    Args:
        parent_weights (str): Path to the pre-trained model weights.
        unfreeze_epoch (int): Epoch at which layers are unfrozen. Default is 9999.
        optimizer (str): Name of the optimizer. Default is 'Adam'.
        scheduler (str): Name of the learning rate scheduler. Default is None.
        scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
        scheduler_interval (str): Scheduler interval. Default is 'epoch'.
        optimizer_args (dict): Arguments for the optimizer. Default is None.
        scheduler_args (dict): Arguments for the scheduler. Default is None.
    """
    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        """
        Add command-line arguments specific to the Graph module.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--parent_weights', type=str, required=True)
        group.add_argument('--frozen_epochs', type=int, default=0)
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')
        return parser
    
    ####################
    # Standard methods #
    ####################
    
    def __init__(self, model, parent_weights, frozen_epochs=0, 
                 optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        """
        Initialize the CNNTransferLearning module.

        Args:
            model (torch.nn.Module): A torch or lightning.pytorch Module.
            parent_weights (str): Path to the pre-trained model weights.
            frozen_epochs (int): Epoch at which layers are unfrozen. Default is 1.
            optimizer (str): Name of the optimizer. Default is 'Adam'.
            scheduler (str): Name of the learning rate scheduler. Default is None.
            scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
            scheduler_interval (str): Scheduler interval. Default is 'epoch'.
            optimizer_args (dict): Arguments for the optimizer. Default is None.
            scheduler_args (dict): Arguments for the scheduler. Default is None.
        """
        super().__init__(model, optimizer, scheduler, scheduler_monitor, 
                         scheduler_interval, optimizer_args, scheduler_args)

        self.parent_weights = parent_weights
        self.frozen_epochs  = frozen_epochs
        
    ###################
    # Non-PTL methods #
    ###################
        
    def attach_parent_weights(self, my_weights):
        """
        Attach parent weights to the model.

        Args:
            my_weights (str): Path to the pre-trained model weights.

        Returns:
            list: List of parameter names that were transferred.
        """
        parent_state_dict = torch.load(my_weights)
        if 'model_state_dict' in parent_state_dict.keys():
            parent_state_dict = parent_state_dict['model_state_dict']
            
        mod_state_dict = filter_state_dict(self.model, parent_state_dict)
        self.model.load_state_dict( mod_state_dict['filtered_state_dict'], strict=False )
        return mod_state_dict['passed_keys']
    
    #############
    # PTL hooks #
    #############
        
    def setup(self, stage='training'):
        """
        Setup method called before training or validation starts.

        Args:
            stage (str): Stage of training. Default is 'training'.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            if 'tar.gz' in self.parent_weights:
                utils.unpack_artifact(self.parent_weights, tmpdirname)
                old_model = utils.model_fn(os.path.join( tmpdirname, 'artifacts' ))
                the_weights = os.path.join( tmpdirname, 'stash_dict.pkl' )
                torch.save(old_model.state_dict(), the_weights)
            elif 'gs://' in self.parent_weights:
                subprocess.call(['gsutil','cp',self.parent_weights,tmpdirname])
                the_weights = os.path.join( tmpdirname, os.path.basename(self.parent_weights) )
            else:
                the_weights = self.parent_weights
            self.transferred_keys = self.attach_parent_weights(the_weights)
        
    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.
        """
        print(f'starting epoch {self.current_epoch}')
        for name, p in self.named_parameters():
            if self.current_epoch < self.frozen_epochs:
                if name in self.transferred_keys:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            else:
                p.requires_grad = True            

class CNNTransferLearningActivityBias(CNNTransferLearning):
    """
    LightningModule for transfer learning with a CNN model and activity bias rebalancing.

    Args:
        model (torch.nn.Module): A torch or lightning.pytorch Module.
        parent_weights (str): Path to the pre-trained model weights.
        frozen_epochs (int): Epoch at which layers are unfrozen. Default is 1.
        rebalance_quantile (float): Quantile value for activity rebalancing. Default is 0.5.
        optimizer (str): Name of the optimizer. Default is 'Adam'.
        scheduler (str): Name of the learning rate scheduler. Default is None.
        scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
        scheduler_interval (str): Scheduler interval. Default is 'epoch'.
        optimizer_args (dict): Arguments for the optimizer. Default is None.
        scheduler_args (dict): Arguments for the scheduler. Default is None.
    """

    ###############################
    # BODA required staticmethods #
    ###############################
    
    @staticmethod
    def add_graph_specific_args(parent_parser):
        """
        Add command-line arguments specific to the Graph module.

        Args:
            parent_parser (argparse.ArgumentParser): Parent argument parser.

        Returns:
            argparse.ArgumentParser: Argument parser with added arguments.
        """
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group  = parser.add_argument_group('Graph Module args')
        group.add_argument('--parent_weights', type=str, required=True)
        group.add_argument('--frozen_epochs', type=int, default=1)
        group.add_argument('--rebalance_quantile', type=float)
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')        
        
        return parser
    
    ####################
    # Standard methods #
    ####################
    
    def __init__(self, model, parent_weights, 
                 rebalance_quantile=0.5, frozen_epochs=1,  
                 optimizer='Adam', scheduler=None, 
                 scheduler_monitor=None, scheduler_interval='epoch', 
                 optimizer_args=None, scheduler_args=None):
        """
        Initialize the CNNTransferLearningActivityBias module.

        Args:
            model (torch.nn.Module): A torch or lightning.pytorch Module.
            parent_weights (str): Path to the pre-trained model weights.
            frozen_epochs (int): Epoch at which layers are unfrozen. Default is 1.
            rebalance_quantile (float): Quantile value for activity rebalancing. Default is 0.5.
            optimizer (str): Name of the optimizer. Default is 'Adam'.
            scheduler (str): Name of the learning rate scheduler. Default is None.
            scheduler_monitor (str): Metric to monitor for the scheduler. Default is None.
            scheduler_interval (str): Scheduler interval. Default is 'epoch'.
            optimizer_args (dict): Arguments for the optimizer. Default is None.
            scheduler_args (dict): Arguments for the scheduler. Default is None.
        """
        super().__init__(model, parent_weights, frozen_epochs, 
                         optimizer, scheduler, scheduler_monitor, 
                         scheduler_interval, optimizer_args, scheduler_args)

        self.rebalance_quantile = rebalance_quantile
        
    #############
    # PTL hooks #
    #############
        
    def setup(self, stage='training'):
        """
        Setup method called before training or validation starts.

        Args:
            stage (str): Stage of training. Default is 'training'.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            if 'gs://' in self.parent_weights:
                subprocess.call(['gsutil','cp',self.parent_weights,tmpdirname])
                the_weights = os.path.join( tmpdirname, os.path.basename(self.parent_weights) )
            else:
                the_weights = self.parent_weights
            self.transferred_keys = self.attach_parent_weights(the_weights)
            
        train_labs = torch.cat([ batch[1].cpu() for batch in self.train_dataloader() ], dim=0)
        
        assert (self.rebalance_quantile > 0.) and (self.rebalance_quantile < 1.)
        
        self.rebalance_cutoff = torch.quantile(train_labs.max(dim=1)[0], self.rebalance_quantile).item()
        print(f"Activity at {self.rebalance_quantile} quantile: {self.rebalance_cutoff}",file=sys.stderr)
        self.upper_factor     = self.rebalance_quantile / (1. - self.rebalance_quantile)
        self.lower_factor     = self.upper_factor ** -1
        
        self.upper_factor = self.upper_factor
        self.lower_factor = self.lower_factor
        
    def training_step(self, batch, batch_idx):
        """
        Perform a training step with activity bias rebalancing.

        Args:
            batch: A batch of data.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: The computed loss value.
        """
        x, y   = batch
        y_hat  = self(x)
        
        get_upper = y.max(dim=1)[0].ge( self.rebalance_cutoff )
        get_lower = ~get_upper
        
        if get_upper.sum():
            loss = self.criterion(y_hat[get_upper], y[get_upper]) \
                     .div(get_upper.numel()).mul(get_upper.sum())\
                     .mul( self.upper_factor )
            if get_lower.sum():
                loss += self.criterion(y_hat[get_lower], y[get_lower]) \
                          .div(get_lower.numel()).mul(get_lower.sum())\
                          .mul( self.lower_factor )
        else:
            loss = self.criterion(y_hat, y) \
                     .mul( self.lower_factor )
        
        self.log('train_loss', loss)
        return loss