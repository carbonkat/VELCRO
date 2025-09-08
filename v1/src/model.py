"""
Model classes and utilities.
"""

"""
Model classes and utilities.
"""

from abc import ABC
from abc import abstractmethod
from typing import Callable, Tuple

import lightning as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loss import Loss
import torch
from torch import nn
from torchmetrics import MeanMetric, MetricCollection
from torchmetrics import MinMetric
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics.classification import BinaryJaccardIndex
import os
import json
from transformers import AutoTokenizer
from transformers import CLIPTokenizer
from transformers import AutoModel

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
    val_loss_best: MinMetric

    def __init__(
        self,
        optimizer: Callable[..., torch.optim.Optimizer],
        scheduler: Callable[..., torch.optim.lr_scheduler.LRScheduler],
        encoder: TwoTowerEncoder,
        loss: Loss,
        data_dir: str,
        kb_path: str,
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
            data_dir (str): Directory where the data is stored.
            kb_path (str): Path to the knowledge base file.
            compile (bool, optional): Whether to compile the model using
                torch.compile, resulting in a speedup and increased memory
                efficiency. The compilation process only works if shapes are
                are consistent across batches, as different shapes will result
                in different computational graphs and further recompilation.
                Defaults to False.
                See: https://pytorch.org/docs/stable/generated/torch.compile.html
        """
        # TODO(liamhebert): We should consider whether we can standardize batches
        # to benefit from torch.compile
        super().__init__()

        # Since net is a nn.Module, it is already saved in checkpoints by
        # default.
        self.save_hyperparameters(logger=False, ignore=["encoder"])
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.encoder = encoder
        self.loss = loss
        self.loss.log = self.log

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        data_dir = os.path.join(self.hparams.data_dir, "v1")
        umls_path = data_dir
        with open(umls_path + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)
        if "openai" in self.encoder.text_model_path:
            text_tokenizer = CLIPTokenizer.from_pretrained(
                self.encoder.text_model_path
            )
        else:
            text_tokenizer = AutoTokenizer.from_pretrained(
                self.encoder.text_model_path
            )

        # We tokenize all the terms together so that we don't have to worry about
        # padding issues when we batch the data. That is, it will automatically
        # pad all the terms to be the same length (the largest sequence).
        umls_text = [x["desc"] for x in umls_terms.values()]
        # For entity description ablations, also have to modify descriptions here
        # TODO: make this a config option
        #caption_ablation = [f'An image of {i.split("[BODY]")[0].split("[TITLE]")[1]}' for i in umls_text]
        #umls_text = caption_ablation
        self.tokenized_umls = text_tokenizer(
            umls_text, return_tensors="pt", padding=True, truncation=True
        )
        del text_tokenizer

        # This makes the dict keys indices and the values the name of the
        # corresponding class. Used for classification metrics.
        reverse_term_dict = {}
        for key in umls_terms.keys():
            reverse_term_dict[umls_terms[key]["idx"]] = key
        self.reverse_term_dict = reverse_term_dict
        num_classes = len(reverse_term_dict.keys())

        # Define metrics
        # TODO: make this more elegant
        self.train_f1 = F1Score(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.train_prec = Precision(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.train_rec = Recall(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.val_f1 = F1Score(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.val_prec = Precision(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.val_rec = Recall(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.test_f1 = F1Score(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.test_prec = Precision(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.test_rec = Recall(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="macro",
        )
        self.label_f1 = F1Score(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="none",
        )

        self.label_precision = Precision(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="none",
        )
        self.label_recall = Recall(
            task="multiclass",
            num_classes=len(reverse_term_dict.keys()),
            average="none",
        )
        self.test_jaccard = BinaryJaccardIndex()
        self.val_jaccard = BinaryJaccardIndex()
        self.train_jaccard = BinaryJaccardIndex()

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

    def model_step(self, batch: dict[str, dict[str, torch.Tensor]]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Compute the loss for a batch of data.

        Args:
            batch (dict): A dict containing two other dicts: one of the label
                data ("y") and one of the input data ("x"). Tensors are expected
                to have a shape of (batch_size, ...).

        Returns:
            The loss value for that batch, using self.loss.
        """
        x, y = batch["x"], batch["y"]
        outs = self.forward(x)
        loss, preds, topk, loss_metrics = self.loss(*outs, y)
        return loss, preds, topk, y, loss_metrics

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()
        self.train_loss.reset()
        self.tokenized_umls.to('cpu')

    def on_test_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.loss.remove_duplicates = False
        self.tokenized_umls.to(self.device)
    
    def on_validation_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.loss.remove_duplicates = False
        self.tokenized_umls.to(self.device)

    def calculate_metrics(self, pred, labels, mode):
        """
        Calculate and log metrics given predictions and true labels.
        Args:
            pred (torch.Tensor): Predictions from the model.
            labels (dict): Dictionary containing true labels and other info.
            mode (str): One of "train", "val", or "test" to indicate the phase.
        """
        # classification metrics
        gts = labels["class_indices"]
        if not self.loss.remove_duplicates:
            batch_preds = pred
        else:
            batch_preds = gts[pred]

        batch_labels = labels["class_indices"]
        correct = torch.eq(batch_preds, batch_labels)
        num_correct = correct.float().sum()
        num_total = batch_labels.shape[0]

        # Log metrics based on the mode
        # TODO: Refactor to reduce redundancy
        if mode == "val":
            self.val_f1(batch_preds, batch_labels)
            self.val_prec(batch_preds, batch_labels)
            self.val_rec(batch_preds, batch_labels)
            self.log(
                f"{mode}/full_precision",
                self.val_prec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_recall",
                self.val_rec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_f1",
                self.val_f1,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
        elif mode == "train":
            self.train_f1(batch_preds, batch_labels)
            self.train_prec(batch_preds, batch_labels)
            self.train_rec(batch_preds, batch_labels)
            self.log(
                f"{mode}/full_precision",
                self.train_prec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_recall",
                self.train_rec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_f1",
                self.train_f1,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
        elif mode == "test":
            self.test_f1(batch_preds, batch_labels)
            self.test_prec(batch_preds, batch_labels)
            self.test_rec(batch_preds, batch_labels)
            self.log(
                f"{mode}/full_precision",
                self.test_prec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_recall",
                self.test_rec,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            self.log(
                f"{mode}/full_f1",
                self.test_f1,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=num_total,
            )
            f1 = self.label_f1(batch_preds, batch_labels)  # [i]
            precision = self.label_precision(batch_preds, batch_labels)  # [i]
            recall = self.label_recall(batch_preds, batch_labels)  # [i]

            for i in range(len(f1)):

                num_label_total = (batch_labels == i).sum()
                self.log(
                    f"{mode}/weight/{self.reverse_term_dict[int(i)]}",
                    num_label_total.to(torch.float32),
                    reduce_fx=sum,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                )
                self.log(
                    f"{mode}/f1/{self.reverse_term_dict[int(i)]}",
                    f1[i],
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=num_label_total,
                    sync_dist=True,
                )
                self.log(
                    f"{mode}/precision/{self.reverse_term_dict[int(i)]}",
                    precision[i],
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=num_label_total,
                    sync_dist=True,
                )
                self.log(
                    f"{mode}/recall/{self.reverse_term_dict[int(i)]}",
                    recall[i],
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=num_label_total,
                    sync_dist=True,
                )

        self.log(
            f"{mode}/unique_batch_labels",
            len(batch_labels.unique()),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=len(batch_labels)
        )
        # accuracy
        acc = num_correct / num_total
        self.log(
            f"{mode}/accuracy",
            acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

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
        loss, preds, topk, targets, loss_metrics = self.model_step(batch)

        # Log extra loss metrics
        for key in loss_metrics.keys():
            self.log(
                f"train/{key}",
                loss_metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        # update and log metrics
        self.train_loss.update(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # If we have mask predictions (len(preds) > 1), also log the IoU
        if len(preds) > 1 and isinstance(preds, tuple):
            mask_preds = preds[1]
            label_preds = preds[0]
            gt_masks = batch['y']['gold_mask']
            self.log(
                "train/iou",
                self.train_jaccard(mask_preds, gt_masks),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            label_preds = preds
        self.calculate_metrics(label_preds, targets, "train")

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
        batch["x"]["candidate_input"] = self.tokenized_umls
        loss, preds, topk, targets, loss_metrics = self.model_step(batch)

        # update and log metrics
        for key in loss_metrics.keys():
            self.log(
                f"val/{key}",
                loss_metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        self.val_loss.update(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # If we have mask predictions (len(preds) > 1), also log the IoU
        if len(preds) > 1 and isinstance(preds, tuple):
            mask_preds = preds[1]
            label_preds = preds[0]
            gt_masks = batch['y']['gold_mask']
            self.log(
                "val/iou",
                self.val_jaccard(mask_preds, gt_masks),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            label_preds = preds

        self.calculate_metrics(label_preds, targets, "val")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best.update(loss)  # update best so far val acc
        # log `val_loss_best` as a value through `.compute()` method, instead of
        # as a metric object otherwise metric would be reset by lightning after
        # each epoch
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            # sync_dist=True,
            prog_bar=True,
        )
        self.loss.remove_duplicates = True

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
        batch["x"]["candidate_input"] = self.tokenized_umls
        loss, preds, topk, targets, loss_metrics = self.model_step(batch)
        for key in loss_metrics.keys():
            self.log(
                f"test/{key}",
                loss_metrics[key],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        self.test_loss.update(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True,
        )

        if len(preds) > 1 and isinstance(preds, tuple):
            mask_preds = preds[1]
            preds = preds[0]
            gt_masks = batch['y']['gold_mask']
            self.log(
                "test/iou",
                self.test_jaccard(mask_preds, gt_masks),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                # sync_dist=True,
            )

        self.calculate_metrics(preds, targets, "test")
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