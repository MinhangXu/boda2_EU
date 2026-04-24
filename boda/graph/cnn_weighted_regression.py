import argparse

import torch

from .cnn_prediction import CNNBasicTraining
from .utils import pearson_correlation, spearman_correlation, shannon_entropy, pearson_r2_score, coefficient_of_determination


class CNNWeightedRegressionTraining(CNNBasicTraining):
    """
    LightningModule for weighted regression training.

    Expected batch formats:
      - (x, y)
      - (x, y, w)

    Sample weights are used only for the loss. Validation / test metrics are
    reported on the raw predictions without weighting so runs remain directly
    comparable across weighted and unweighted training.
    """

    @staticmethod
    def add_graph_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        group = parser.add_argument_group('Graph Module args')
        group.add_argument('--optimizer', type=str, default='Adam')
        group.add_argument('--scheduler', type=str)
        group.add_argument('--scheduler_monitor', type=str)
        group.add_argument('--scheduler_interval', type=str, default='epoch')
        group.add_argument('--weighted_loss_reduction', type=str, default='mean', choices=['mean'])
        return parser

    @staticmethod
    def process_args(grouped_args):
        graph_args = super(CNNWeightedRegressionTraining, CNNWeightedRegressionTraining).process_args(grouped_args)
        return graph_args

    def __init__(self, model, optimizer='Adam', scheduler=None,
                 scheduler_monitor=None, scheduler_interval='epoch',
                 optimizer_args=None, scheduler_args=None,
                 weighted_loss_reduction='mean'):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_monitor=scheduler_monitor,
            scheduler_interval=scheduler_interval,
            optimizer_args=optimizer_args,
            scheduler_args=scheduler_args,
        )
        self.weighted_loss_reduction = weighted_loss_reduction

    def _unpack_batch(self, batch):
        if len(batch) == 2:
            x, y = batch
            w = None
        elif len(batch) == 3:
            x, y, w = batch
        else:
            raise ValueError(f'Unexpected batch length: {len(batch)}')
        return x, y, w

    def _align_shapes(self, y_hat, y):
        if y_hat.dim() == 2 and y_hat.shape[1] == 1 and y.dim() == 1:
            y_hat = y_hat.squeeze(1)
        if y.dim() == 2 and y.shape[1] == 1 and y_hat.dim() == 1:
            y = y.squeeze(1)
        return y_hat, y

    def _weighted_mse(self, y_hat, y, w):
        y_hat, y = self._align_shapes(y_hat, y)
        if w is None:
            return self.criterion(y_hat, y)

        if w.dim() > 1:
            w = w.reshape(-1)
        per_sample = (y_hat - y).pow(2)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        w = w.to(per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-8)

    def training_step(self, batch, batch_idx):
        x, y, w = self._unpack_batch(batch)
        y_hat = self(x)
        loss = self._weighted_mse(y_hat, y, w)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = self._unpack_batch(batch)
        y_hat = self(x)
        y_hat, y = self._align_shapes(y_hat, y)

        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss)

        step_valid_pearson_r2 = pearson_r2_score(y, y_hat)
        self.log('step_valid_pearson_r2', step_valid_pearson_r2)

        pearsonr, mean_pearsonr = pearson_correlation(y_hat, y)
        self.log('valid_mean_pearson', mean_pearsonr)

        if y_hat.dim() == 2 and y_hat.shape[1] == 3:
            cell_types = ['K562', 'HepG2', 'SKNSH']
            for i, coeff in enumerate(pearsonr):
                self.log(f'valid_pearson_{cell_types[i]}', coeff)
                self.log(f'valid_pearson_squared_{cell_types[i]}', coeff ** 2)

        metric = self.categorical_mse(y_hat, y)
        return {'loss': loss, 'metric': metric, 'preds': y_hat, 'labels': y, 'pearson_r2': step_valid_pearson_r2}

    def validation_epoch_end(self, val_step_outputs):
        arit_mean = torch.stack([batch['loss'] for batch in val_step_outputs], dim=0).mean()
        harm_mean = torch.stack([batch['metric'] for batch in val_step_outputs], dim=0).mean(dim=0).pow(-1).mean().pow(-1)

        epoch_preds = torch.cat([batch['preds'] for batch in val_step_outputs], dim=0)
        epoch_labels = torch.cat([batch['labels'] for batch in val_step_outputs], dim=0)

        spearman, mean_spearman = spearman_correlation(epoch_preds, epoch_labels)
        shannon_pred, shannon_label = shannon_entropy(epoch_preds), shannon_entropy(epoch_labels)
        _, specificity_mean_spearman = spearman_correlation(shannon_pred, shannon_label)
        val_pearson_r2 = pearson_r2_score(epoch_labels, epoch_preds)
        val_cod_r2 = coefficient_of_determination(epoch_labels, epoch_preds)

        metrics = {
            'current_epoch': self.current_epoch,
            'arithmetic_mean_loss': arit_mean,
            'harmonic_mean_loss': harm_mean,
            'prediction_mean_spearman': mean_spearman.item(),
            'entropy_spearman': specificity_mean_spearman.item(),
            'epoch_end_val_pearson_r2': val_pearson_r2,
            'epoch_end_val_cod_r2': val_cod_r2,
            # Normalized summary keys shared with CNNBasicTraining so that
            # `train_wandb_log.build_provenance_record` can read a single set
            # of keys regardless of which graph module ran the training.
            'val_pearson_r2': val_pearson_r2,
            'val_cod_r2': val_cod_r2,
            'val_loss': arit_mean,
            'val_pearson': pearson_correlation(epoch_preds, epoch_labels)[1].item(),
            'val_spearman': mean_spearman.item(),
        }
        self.aug_log(external_metrics=metrics)
        return None

    def test_step(self, batch, batch_idx):
        """
        Unpack the optional barcode weight and mirror `validation_step`. The
        weighted loss is only applied at train time; test metrics use the
        unweighted criterion so they are directly comparable across
        basic/weighted training regimes.
        """
        x, y, _ = self._unpack_batch(batch)
        y_hat = self(x)
        y_hat, y = self._align_shapes(y_hat, y)
        loss = self.criterion(y_hat, y)
        self.log('step_test_loss', loss, on_epoch=True)
        return {'loss': loss.detach(), 'preds': y_hat.detach(), 'labels': y.detach()}

    def test_epoch_end(self, test_step_outputs):
        """
        Same aggregation contract as `CNNBasicTraining.test_epoch_end`: log
        test_loss / test_pearson_r2 / test_cod_r2 / test_pearson /
        test_spearman to W&B summary so `runs.csv` can persist final
        held-out metrics.
        """
        if not test_step_outputs:
            return None

        loss = torch.stack([b['loss'] for b in test_step_outputs], dim=0).mean()
        epoch_preds = torch.cat([b['preds'] for b in test_step_outputs], dim=0)
        epoch_labels = torch.cat([b['labels'] for b in test_step_outputs], dim=0)

        test_pearson_r2 = pearson_r2_score(epoch_labels, epoch_preds)
        test_cod_r2 = coefficient_of_determination(epoch_labels, epoch_preds)
        _, mean_pearson = pearson_correlation(epoch_preds, epoch_labels)
        _, mean_spearman = spearman_correlation(epoch_preds, epoch_labels)

        on_epoch = True
        self.log('test_loss', loss, on_epoch=on_epoch)
        self.log('test_pearson_r2', test_pearson_r2, on_epoch=on_epoch)
        self.log('test_cod_r2', test_cod_r2, on_epoch=on_epoch)
        self.log('test_pearson', mean_pearson.item() if hasattr(mean_pearson, 'item') else float(mean_pearson), on_epoch=on_epoch)
        self.log('test_spearman', mean_spearman.item() if hasattr(mean_spearman, 'item') else float(mean_spearman), on_epoch=on_epoch)
        self.log('epoch_end_test_pearson_r2', test_pearson_r2, on_epoch=on_epoch)
        self.log('epoch_end_test_cod_r2', test_cod_r2, on_epoch=on_epoch)
        return None
