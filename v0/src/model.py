## our contrastive learning model
# right now its just a framework with some base code we can switch around

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CosineSimilarity
from pytorch_metric_learning import losses
from transformers import CLIPModel, AutoModel
import os
import json
from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    AutoTokenizer,
)


class mmContrastiveModel(pl.LightningModule):

    def __init__(
        self,
        model_path="openai/clip-vit-base-patch16",
        mask_kernel="square",
        umls_dir: str = "data",
    ):
        super().__init__()
        self.save_hyperparameters()
        """
        Here we put contrastive learning stuff
        """
        if "openai" in model_path:
            self.visionmodel = CLIPVisionModelWithProjection.from_pretrained(
                model_path, local_files_only=False
            )
            self.vision_projection = self.visionmodel.visual_projection
        else:
            self.visionmodel = AutoModel.from_pretrained(
                model_path, local_files_only=False
            )
            self.vision_projection = nn.Linear(1024, 768)

        # self.expand_embedding_blowup_mask = nn.Conv2d(3, 1, (32, 32), (32, 32), bias=False)
        self.similarity = CosineSimilarity(dim=-1)
        # self.image_projection = nn.Linear(1024, 768)
        self.loss = nn.CrossEntropyLoss(reduction="mean")
        kernel = torch.zeros((14, 14))
        if mask_kernel == "square":

            mid_point = 7
            mid_mid_point = mid_point // 2

            # kernel[mid_point - mid_mid_point:mid_point + mid_mid_point, mid_point - mid_mid_point:mid_point + mid_mid_point] = 1
            # kernel still doing zeroing out imgs so im making it all ones for now.
            kernel = torch.ones((14, 14))
        elif mask_kernel == "cross":
            mid_point = 7
            mid_mid_point = mid_point // 2
            kernel[mid_point - mid_mid_point : mid_point + mid_mid_point, :] = 1
            kernel[:, mid_point - mid_mid_point : mid_point + mid_mid_point] = 1

        kernel = kernel.tile(3, 1, 1).unsqueeze(0)

        # load umls terms!
        # we will probably want to generate these ahead of time and load
        with open(umls_dir + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)
            umls_values = []
            umls_ids = {}
            for k, v in umls_terms.items():
                umls_values.append(k)  # term name
                umls_ids[int(v["idx"])] = k  # term id
            # umls_values = [x['desc'] for x in umls_terms.keys()]
            # umls_terms = {int(x['idx']): k for k, x in umls_terms.items()}
            umls_terms = umls_ids
            # umls_terms.append("background")
            if "openai" in model_path:
                tokenizer = CLIPTokenizer.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # this needs to be done but is causing an error
            # tokenizer.add_special_tokens({"additional_special_tokens": ["[TITLE]", "[BODY]"]})
            self.umls_terms = umls_terms
            self.umls_tokens = tokenizer(
                umls_values, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            if self.umls_tokens.input_ids.requires_grad:
                raise Exception("UMLS tokens requires grad!")
            del tokenizer
        if "openai" in model_path:
            self.textmodel = CLIPTextModelWithProjection.from_pretrained(
                model_path
            ).to(self.device)
            self.size = int(
                model_path.split("patch")[1]
            )  # assume last few digits is the patch size
        else:
            self.textmodel = AutoModel.from_pretrained("bert-base-uncased").to(
                self.device
            )
            self.size = 16
        self.is_bert = "openai" not in model_path

        mid_point = self.size // 2
        mid_mid_point = mid_point // 2

        # kernel[mid_point - mid_mid_point:mid_point + mid_mid_point, mid_point - mid_mid_point:mid_point + mid_mid_point] = 1
        # kernel still doing zeroing out imgs so im making it all ones for now.
        kernel = torch.ones((self.size, self.size))

        kernel = kernel.tile(1, 3, 1, 1)
        self.expand_embedding_blowup_mask = nn.Conv2d(
            3, 1, (self.size, self.size), (self.size, self.size), bias=False
        )
        self.expand_embedding_blowup_mask.requires_grad = False
        self.expand_embedding_blowup_mask.weight = nn.Parameter(
            kernel, requires_grad=False
        )

        self.training_output_preds = []
        self.training_output_labels = []
        self.training_output_loss = []

        self.val_output_preds = []
        self.val_output_labels = []
        self.val_output_loss = []

        self.test_output_preds = []
        self.test_output_labels = []
        self.test_output_loss = []

    def mod_loss(self, y, roi_embeddings, umls_embeddings):
        # we need to project the mask embeddings into the same dim as umls (batch x 768)
        # the clip projection head looks more complicated than this so idk if we need to add some other fancy stuff on top
        # projected_masks = self.vision_projection(roi_embeddings) # these are the mask embe
        # print(umls_embeddings)
        # print(roi_embeddings)
        # print('LOSS', y.argmax(dim=-1).shape, roi_embeddings.shape, umls_embeddings.shape)

        probs = (
            self.similarity(roi_embeddings, umls_embeddings.unsqueeze(1)) * 20
        ).T
        # print('sims', sims.shape)
        # probs = F.softmax(sims, dim=-1)

        loss = self.loss(probs, y)
        # print(loss)
        # print(loss)
        # IT WOOOOOORKSSSS!!!!
        return probs, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-6)
        print("GOT STEPS", self.trainer.max_steps)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": torch.optim.lr_scheduler.PolynomialLR(
                optimizer, power=1, total_iters=self.trainer.max_steps
            ),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def forward(self, x, mask):
        masked_embeds_list = []

        imgs = x["pixel_values"].squeeze(1)
        mask = (self.expand_embedding_blowup_mask(mask) > 0).float()
        # y = torch.flatten(x['input_ids'], start_dim=0,end_dim=1)
        # y = [unstack[i] for i in range(len(unstack)) if i not in dummy_indices]
        # y = torch.stack(y)

        x = self.visionmodel(imgs)
        image_embeds = x.last_hidden_state
        image_embeds = image_embeds[:, 1:, :]
        image_embeds = self.vision_projection(image_embeds)

        mask = mask.flatten(start_dim=1).unsqueeze(-1).float()
        # print(mask.shape, image_embeds.shape)
        masked_embeds = torch.matmul(
            image_embeds.transpose(-2, -1), mask
        ).squeeze(
            -1
        )  # this is just the sum
        # need to take the mean now, scale by the amount of trues in the mask
        # print('SECOND', mask.shape, masked_embeds.shape)
        # print(mask.squeeze(-1).sum(dim=-1))
        masked_embeds = masked_embeds / mask.squeeze(-1).sum(dim=-1).unsqueeze(
            -1
        )
        # print('res', masked_embeds.shape)
        # remove dummys from masked_embeds_list too!
        self.umls_tokens = self.umls_tokens.to(self.device)
        if self.is_bert:
            umls_embeddings = self.textmodel(
                **self.umls_tokens, return_dict=True
            ).last_hidden_state[:, 0, :]
        else:
            umls_embeddings = self.textmodel(**self.umls_tokens).text_embeds
        return masked_embeds, umls_embeddings

    def step(self, batch, mode="train"):
        data, mask, labels, _ = batch
        masked_embeds, umls_embeds = self(data, mask)
        probs, loss = self.mod_loss(labels, masked_embeds, umls_embeds)
        # self.calculate_metrics(probs.detach(), labels.detach(), mode)
        return masked_embeds, umls_embeds, loss, probs

    def calculate_metrics(self, pred, labels, loss, mode):
        # classification metrics
        batch_preds = pred.argmax(dim=-1)
        batch_labels = labels
        correct = batch_preds == batch_labels
        num_correct = correct.float().sum()

        for i in batch_labels.unique():
            num_label_correct = torch.logical_and(
                (batch_preds == batch_labels), (batch_labels == i)
            ).sum()
            num_label_total = (batch_labels == i).sum()
            num_label_pred = (batch_preds == i).sum()
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
            self.log(
                f"{mode}/f1/{self.umls_terms[int(i)]}",
                f1,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=num_label_total,
                sync_dist=True,
            )
            self.log(
                f"{mode}/precision/{self.umls_terms[int(i)]}",
                precision,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=num_label_total,
                sync_dist=True,
            )
            self.log(
                f"{mode}/recall/{self.umls_terms[int(i)]}",
                recall,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=num_label_total,
                sync_dist=True,
            )

        num_total = batch_labels.shape[0]

        # accuracy
        acc = num_correct / num_total
        # print('LOSSES', loss, '\n\n\n')
        self.log(
            f"{mode}/accuracy",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/loss",
            torch.mean(loss),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        # precision

        ...

    def training_step(self, batch, batch_idx):
        masked_embeds, umls_embeds, loss, probs = self.step(batch, mode="train")

        if loss.detach().isnan():
            print(
                "GOT NAN IGNORING", loss.detach(), probs, batch[2], flush=True
            )
            print("IMG", torch.any(torch.isnan(batch[0]["pixel_values"])))
            print("MASK", torch.any(torch.isnan(batch[1])))
            print("MASK_MAX", torch.max(batch[1].flatten(start_dim=1)))
        else:
            self.log(
                "train/loss_step",
                loss.detach(),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            self.training_output_preds.append(probs.detach().cpu())
            self.training_output_labels.append(batch[2].detach().cpu())
            self.training_output_loss.append(loss.detach().cpu())
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.training_output_preds, dim=0)
        all_labels = torch.cat(self.training_output_labels, dim=0)
        loss = torch.stack(self.training_output_loss, dim=0)
        self.calculate_metrics(all_preds, all_labels, loss, "train")
        self.training_output_labels.clear()
        self.training_output_preds.clear()
        self.training_output_loss.clear()

    def validation_step(self, batch, batch_idx):
        # print("validation step")
        masked_embeds, umls_embeds, loss, probs = self.step(batch, mode="val")
        # self.log("val/loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=False)
        if loss.detach().isnan():
            print(
                "GOT NAN IGNORING", loss.detach(), probs, batch[2], flush=True
            )
            print("IMG", torch.any(torch.isnan(batch[0]["pixel_values"])))
            print("MASK", torch.any(torch.isnan(batch[1])))
            print("MASK_MAX", torch.max(batch[1].flatten(start_dim=1)))
        else:
            self.val_output_preds.append(probs.detach().cpu())
            self.val_output_labels.append(batch[2].detach().cpu())
            self.val_output_loss.append(loss.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_output_preds, dim=0)
        all_labels = torch.cat(self.val_output_labels, dim=0)
        loss = torch.stack(self.val_output_loss, dim=0)
        self.calculate_metrics(all_preds, all_labels, loss, "val")
        self.val_output_labels.clear()
        self.val_output_preds.clear()
        self.val_output_loss.clear()

    def test_step(self, batch, batch_idx):
        # print("test step")
        masked_embeds, umls_embeds, loss, probs = self.step(batch, mode="test")

        # self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        if loss.detach().isnan():
            print("GOT NAN IGNORING", loss.detach(), probs, batch[2])
        else:
            self.test_output_preds.append(probs.detach().cpu())
            self.test_output_labels.append(batch[2].detach().cpu())
            self.test_output_loss.append(loss.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.cat(self.test_output_preds, dim=0)
        all_labels = torch.cat(self.test_output_labels, dim=0)
        loss = torch.stack(self.test_output_loss, dim=0)
        self.calculate_metrics(all_preds, all_labels, loss, "test")
        self.test_output_labels.clear()
        self.test_output_preds.clear()
        self.test_output_loss.clear()

        # loss = self.loss(label_logit, labels)

        # self.log("test/loss", loss)
