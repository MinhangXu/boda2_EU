from .cnn_prediction import CNNBasicTraining


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
        parser = CNNBasicTraining.add_graph_specific_args(parent_parser)
        group = parser.add_argument_group('Weighted regression args')
        group.add_argument('--weighted_loss_reduction', type=str, default='mean', choices=['mean'])
        return parser

    @staticmethod
    def process_args(grouped_args):
        return CNNBasicTraining.process_args(grouped_args)

    def __init__(self, model, optimizer='Adam', scheduler=None,
                 scheduler_monitor=None, scheduler_interval='epoch',
                 optimizer_args=None, scheduler_args=None,
                 weighted_loss_reduction='mean', output_names=None,
                 log_per_output_metric_details=True,
                 log_legacy_metric_aliases=True):
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_monitor=scheduler_monitor,
            scheduler_interval=scheduler_interval,
            optimizer_args=optimizer_args,
            scheduler_args=scheduler_args,
            output_names=output_names,
            log_per_output_metric_details=log_per_output_metric_details,
            log_legacy_metric_aliases=log_legacy_metric_aliases,
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

    def _weighted_mse(self, y_hat, y, w):
        y_hat, y = self.align_prediction_and_label_shapes(y_hat, y)
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
        if self.log_legacy_metric_aliases:
            self.log('train_loss', loss)
        y_hat, y = self.align_prediction_and_label_shapes(y_hat, y)
        return {'loss': loss, 'preds': y_hat.detach(), 'labels': y.detach()}
