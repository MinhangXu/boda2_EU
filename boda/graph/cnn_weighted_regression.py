import torch

from .cnn_prediction import CNNBasicTraining, CNNBassetBranchedScopedTransfer


class _StrictWeightedMSEMixin:
    """Use one finite, nonnegative scalar weight for every training example.

    Weighted graph classes are deliberately fail-closed: selecting a weighted
    graph while the data module returns an unweighted batch is a provenance
    error, not a request to fall back to ordinary MSE.  Validation and test
    behavior remain inherited from ``CNNBasicTraining`` and therefore ignore
    the optional third batch tensor.
    """

    @staticmethod
    def _unpack_weighted_batch(batch):
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
            observed = len(batch) if isinstance(batch, (list, tuple)) else type(batch).__name__
            raise ValueError(
                "Strict weighted training requires a three-item (x, y, w) "
                f"batch; observed {observed!r}"
            )
        return batch[0], batch[1], batch[2]

    def _weighted_mse(self, y_hat, y, w):
        y_hat, y = self.align_prediction_and_label_shapes(y_hat, y)
        if not torch.is_tensor(w):
            raise TypeError("Training sample weights must be a torch.Tensor")

        batch_size = int(y.shape[0])
        weights = w.reshape(-1)
        if int(weights.numel()) != batch_size:
            raise ValueError(
                "Strict weighted training requires exactly one weight per "
                f"sample; observed {weights.numel()} weights for {batch_size} samples"
            )
        weights = weights.to(device=y_hat.device, dtype=y_hat.dtype)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("Training sample weights must all be finite")
        if bool((weights < 0).any()):
            raise ValueError("Training sample weights must be nonnegative")
        weight_sum = weights.sum()
        if not bool(torch.isfinite(weight_sum)) or float(weight_sum.detach().cpu()) <= 0:
            raise ValueError("Training sample weights must have a positive finite sum")

        per_sample = (y_hat - y).pow(2)
        if per_sample.dim() > 1:
            per_sample = per_sample.mean(dim=1)
        return (per_sample * weights).sum() / weight_sum

    def training_step(self, batch, batch_idx):
        x, y, w = self._unpack_weighted_batch(batch)
        y_hat = self(x)
        loss = self._weighted_mse(y_hat, y, w)
        if self.log_legacy_metric_aliases:
            self.log('train_loss', loss)
        y_hat, y = self.align_prediction_and_label_shapes(y_hat, y)
        return {'loss': loss, 'preds': y_hat.detach(), 'labels': y.detach()}


class CNNWeightedRegressionTraining(_StrictWeightedMSEMixin, CNNBasicTraining):
    """
    LightningModule for weighted regression training.

    Expected training batch format:
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
        graph_args = CNNBasicTraining.process_args(grouped_args)
        weighted_args = grouped_args.get('Weighted regression args')
        if weighted_args is not None:
            graph_args.weighted_loss_reduction = weighted_args.weighted_loss_reduction
        return graph_args

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



class CNNBassetBranchedScopedWeightedTransfer(
    _StrictWeightedMSEMixin, CNNBassetBranchedScopedTransfer
):
    """Barcode-weighted counterpart of the frozen scoped Enhancer transfer.

    The parent-artifact verification, selected-head loading, warm-up/unfreeze
    policy, differential learning rates, and optimizer-state reset are
    inherited unchanged.  Only the training loss is replaced.
    """
