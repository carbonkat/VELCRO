"""
Model classes and utilities.
"""

from abc import ABC
from abc import abstractmethod
from typing import Callable, Tuple

from loss import Loss
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
from torch import nn
from torchmetrics import MaxMetric
from torchmetrics import MeanMetric


class TwoTowerEncoder(nn.Module, ABC):
    """
    Generic encoder class that should be implemented by experimental models.
    """

    @abstractmethod
    def forward(
        self,
        candidate_input: dict[str, torch.Tensor],
        image_input: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates the required embeddings for the text and image inputs.

        Args:
            candidate_input (dict): Dict of the inputs required for the candidate
                tower.
            image_input (dict): Dict of the inputs required for the image tower.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Returns two tensors, the first
            representing the candidate embeddings and the second representing the
            image embeddings.
        """
        ...


class Model(pl.LightningModule):
    """
    Base class for models, fitting the PyTorch Lightning interface.
    """

    encoder: TwoTowerEncoder
    loss: Loss

    train_loss: MeanMetric
    val_loss: MeanMetric
    test_loss: MeanMetric
    val_loss_best: MaxMetric

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        encoder: TwoTowerEncoder,
        loss: Loss,
        compile: bool = False,
    ) -> None:
        """Initializes the model.

        Args:
            optimizer (Callable[..., torch.optim.Optimizer]): Callable to create
                the model optimizer. The callable should have all fields set
                except for the `params` field.
            scheduler (Callable[..., torch.optim.lr_scheduler.LRScheduler]):
                Callable to create the model scheduler. The callable should have
                all fields set except for the `optimizer` field.
            encoder (TwoTowerEncoder): The main guts of the model, which will
                generate the embeddings for the candidate item and image inputs.
            loss (Loss): Loss that can take in roi and candidate embeddings
                with true alignment indices.
            compile (bool, optional): Whether to compile the model using torch.
                This only works if shapes are consistent across batches.
                Defaults to False.
        """
        super().__init__()

        # Since net is a nn.Module, it is already saved in checkpoints by
        # default.
        self.save_hyperparameters(logger=False, ignore=["encoder"])

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.encoder = encoder
        self.loss = loss

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MaxMetric()

    # TODO(liamhebert): Implement model logic

    def forward(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the forward pass of the model.

        Args:
            x (dict): The input data, to be mapped to the model. It is expected
                that each tensor has a shape of (batch_size, ...).

        Returns:
            Two tensors, the first representing the roi embeddings, the second
            representing the candidate embeddings.
        """
        return self.encoder(**x)

    def model_step(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the loss for a batch of data.

        Args:
            batch (dict): A dict containing two other dicts: one of the label
                data ("y") and one of the input data ("x"). Tensors are expected
                to have a shape of (batch_size, ...).

        Returns:
            The loss value for that batch, using self.loss.
        """
        x, y = batch["x"], batch["y"]
        roi_embeddings, candidate_embeddings = self.forward(x)
        loss, preds = self.loss(roi_embeddings, candidate_embeddings, y)

        return loss, preds, y

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def training_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss and metrics for a training batch of data.

        Args:
            batch (dict): A dict containing two other dicts: one of the label
                data ("y") and one of the input data ("x"). Tensors are expected
                to have a shape of (batch_size, ...).
            batch_idx (int): The index of the batch.

        Returns:
            The loss value for that batch, using self.loss.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # return loss or backpropagation will fail
        return loss

    def validation_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int
    ) -> None:
        """Compute the loss and metrics for a validation batch of data.

        Args:
            batch (dict): A dict containing two other dicts: one of the label
                data ("y") and one of the input data ("x"). Tensors are expected
                to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(
        self, batch: dict[str, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Compute the loss and metrics for a test batch of data.

        Args:
            batch (dict): A dict containing two other dicts: one of the label
                data ("y") and one of the input data ("x"). Tensors are expected
                to have a shape of (batch_size, ...).
            batch_idx: The index of the batch.

        Returns:
            The loss value for that batch, using self.loss.
        """
        loss, preds, targets = self.model_step(batch)

        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":  # type: ignore
            self.encoder = torch.compile(self.encoder)  # type: ignore

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate
            schedulers to be used for training.
        """
        assert self.trainer.model is not None

        optimizer = self.hparams.optimizer(  # type: ignore
            params=self.trainer.model.parameters()
        )

        if self.hparams.scheduler is not None:  # type: ignore
            scheduler = self.hparams.scheduler(  # type: ignore
                optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
