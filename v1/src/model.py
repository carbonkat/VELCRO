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
from torchmetrics import MeanMetric
from torchmetrics import MinMetric
from torchmetrics import F1Score
from torchmetrics import Precision
from torchmetrics import Recall
import os
import json
from transformers import AutoTokenizer
from transformers import CLIPTokenizer
from transformers import AutoModel

"""
from torchmetrics import Metric

class MyWeight(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("weight", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, weight) -> None:
        weight = self._input_format(weight)
        self.weight += weight

    def compute(self) -> torch.Tensor:
        return self.weight
"""


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
        print(text_tokenizer)
        # We tokenize all the terms together so that we don't have to worry about
        # padding issues when we batch the data. That is, it will automatically
        # pad all the terms to be the same length (the largest sequence).
        umls_text = [x["desc"] for x in umls_terms.values()]
        self.tokenized_umls = text_tokenizer(
            umls_text, return_tensors="pt", padding=True, truncation=True
        )
        # self.tokenized_umls.to(device)
        # print(self.tokenized_umls['input_ids'].shape)

        # This makes the dict keys indices and the values the name of the
        # corresponding class. Used for classification metrics.
        reverse_term_dict = {}
        for key in umls_terms.keys():
            reverse_term_dict[umls_terms[key]["idx"]] = key
        self.reverse_term_dict = reverse_term_dict

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

        # self.testing_output_preds = []
        # self.testing_output_labels = []

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
        outs = self.forward(x)
        # print("candidate embed shape", outs[0].shape)
        loss, preds = self.loss(*outs, y)
        # (loss)
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

    # def on_validation_epoch_start(self) -> None:
    #    print("validation start gothere!")
    #    self.loss.remove_duplicates=False
    #    self.tokenized_umls.to(self.device)

    def on_test_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.loss.remove_duplicates = False
        # self.tokenized_umls.to(self.device)

    """
    #def on_test_epoch_end(self) -> None:
        all_preds = torch.cat(self.testing_output_preds, dim=0)
        all_labels = torch.cat(self.testing_output_labels, dim=0)
        mode = "test"
        all_tp = 0
        all_tp_fp = 0
        all_tp_fn = 0
        for i in all_labels.unique():
            num_label_correct = torch.logical_and(
                (all_preds == all_labels), (all_labels == i)
            ).sum()
            num_label_total = (all_labels == i).sum()
            num_label_pred = (all_preds == i).sum()
            if (
                num_label_correct == 0
                or num_label_pred == 0
                or num_label_total == 0
            ):
                precision = torch.tensor(0.0).float()
                recall = torch.tensor(0.0).float()
                f1 = torch.tensor(0.0).float()
            else:
                precision = num_label_correct / num_label_pred
                recall = num_label_correct / num_label_total
                f1 = 2 * (precision * recall) / (precision + recall)
            all_tp += num_label_correct
            all_tp_fp += num_label_pred
            all_tp_fn += num_label_total

        all_precision = all_tp / all_tp_fp
        all_recall = all_tp / all_tp_fn
        all_f1 = 2 * (all_precision * all_recall) / (all_precision + all_recall)
        self.log(
            f"{mode}/man_calc_f1_full",
            all_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/man_calc_precision_full",
            all_precision,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/man_calc_recall_full",
            all_recall,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
    """

    def calculate_metrics(self, pred, labels, mode):
        """
        docstring here
        """
        # classification metrics
        gts = labels["class_indices"]
        # batch_preds = gts[pred]
        if not self.loss.remove_duplicates:
            batch_preds = pred
        else:
            batch_preds = gts[pred]

        batch_labels = labels["class_indices"]
        correct = torch.eq(batch_preds, batch_labels)
        num_correct = correct.float().sum()
        # self.f1(batch_preds, batch_labels)
        # self.recall(batch_preds, batch_labels)
        # self.precision(batch_preds, batch_labels)
        num_total = batch_labels.shape[0]

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
            # print("before batchifying!", pred)
            # print("after batchifying!", batch_preds)
            # print("ground truths!", gts)
        #    self.testing_output_preds.append(batch_preds)
        #    self.testing_output_labels.append(batch_labels)
        """
        self.log(
            f"{mode}/full_precision",
            self.precision,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=num_total,
            #sync_dist=True,
        )
        self.log(
            f"{mode}/full_recall",
            self.recall,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=num_total,
            #sync_dist=True,
        )
        self.log(
            f"{mode}/full_f1",
            self.f1,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=num_total,
            #sync_dist=True,
        )
        """
        self.log(
            f"{mode}/unique_batch_labels",
            len(batch_labels.unique()),
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        """
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
                # batch_size=num_label_total,
                #sync_dist=True,
            )
            self.log(
                f"{mode}/f1/{self.reverse_term_dict[int(i)]}",
                f1[i],
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=num_label_total,
                sync_dist=True,
            )
            self.log(
                f"{mode}/precision/{self.reverse_term_dict[int(i)]}",
                precision[i],
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=num_label_total,
                sync_dist=True,
            )
            self.log(
                f"{mode}/recall/{self.reverse_term_dict[int(i)]}",
                recall[i],
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=num_label_total,
                sync_dist=True,
            )
            """
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
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.calculate_metrics(preds, targets, "train")

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
        # print("validation before model step!")
        self.loss.remove_duplicates = False
        batch["x"]["candidate_input"] = self.tokenized_umls.to(self.device)
        loss, preds, targets = self.model_step(batch)
        # print(preds.shape)
        # print(preds)
        # print("validation after model step!")
        # update and log metrics
        self.val_loss(loss)
        self.log(
            "val/loss",
            self.val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # sync_dist=True,
        )
        # print("validation before calculating metrics!")
        self.calculate_metrics(preds, targets, "val")

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
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.log(
            "test/loss",
            self.test_loss,
            on_step=False,
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
