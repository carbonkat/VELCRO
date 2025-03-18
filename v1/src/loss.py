"""
Loss classes.
"""

from abc import ABC
from abc import abstractmethod

import torch
from torch import nn
from monai.losses import DiceLoss
from monai.losses import FocalLoss
from typing import Callable


class Loss(ABC, nn.Module):
    """
    Abstract class for loss functions.
    """

    log: Callable

    @abstractmethod
    def __call__(
        self,
        roi_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        y_true: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            roi_embeddings: Predicted embeddings for each region of interest.
                It is expected that each roi in all the images are flattened with
                shape (batch_size*num_rois, embedding_size)
            candidate_embeddings: Predicted embeddings for each candidate item.
                It is expected that all potential candidates are flattened with
                shape (num_candidates, embedding_size).
            y_true: The true values.

        Returns:
            The loss value and the predicted values.
        """
        pass


class ContrastiveLoss(Loss):
    """Contrastive loss function between the roi and candidate embeddings using
    in-batch negatives.

    This is done by aligning the positive roi regions to the positive candidate
    embedding, and the treating all other candidate embeddings as negatives. The
    implementation is similar to a cross entropy loss, where the "probability
    logits" of each class (candidates) is the cosine similarity score between the
    roi and candidate embeddings.

    This implementation is as proposed by InfoNCE, but with a modification that
    handles duplicate positive pairs.

    Since we use in-batch negatives, it is possible that multiple items within
    the same batch have the same positive candidate. However, InfoNCE only works
    with a single positive class (due to cross entropy loss). To handle this, we
    have an optional parameter ("remove_duplicates") that will check for and then
    remove duplicate positive and negative pairs.

    See: https://paperswithcode.com/method/infonce for more details.
    """

    cosine_similarity: nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=2)
    remove_duplicates: bool = True
    temperature: nn.Parameter

    def __init__(
        self,
        remove_duplicates: bool = True,
        temperature: float = 0.30,
        learnable_temperature: bool = False,
    ):
        """Initializes the contrastive loss.

        Args:
            remove_duplicates (bool, optional): Whether to remove duplicate values
                from the loss calculation. If you are guaranteed to not have
                duplicate candidates, then this should be set to False for a
                speed up. Defaults to True.
            temperature (float, optional): The temperature to use for the softmax
                function. A higher value will make the distribution more uniform,
                while a lower value will make the distribution more peaky.
                Defaults to 0.30.
            learnable_temperature (bool, optional): Whether to learn the
                temperature parameter. The initial value of the temperature will
                be `temperature`. Defaults to False.
        """
        super().__init__()
        self.remove_duplicates = remove_duplicates
        self.temperature = nn.Parameter(
            torch.tensor([temperature]), requires_grad=learnable_temperature
        )

    def __call__(
        self,
        roi_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        y_true: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the cross-entropy loss.

        Args:
            roi_embeddings: The embeddings of each region of interest, with
                shape (batch_size * num_rois, embedding_size).
            candidate_embeddings: The embeddings for each candidate item, with
                shape (num_candidates, embedding_size) if remove_duplicates is
                false, (batch_size * num_rois, embedding_size) otherwise.
            y_true: A dictionary containing the field "class_indices", which is a
                tensor of shape (batch_size * num_rois) containing the indices of
                the true class for each roi.

        Returns:
            The cross-entropy loss value.
        """
        # print("roi embeddings shape", roi_embeddings.shape)
        # print("candidate text shape", candidate_embeddings.shape)
        class_indices = y_true["class_indices"]
        flattened_batch, roi_embed_dim = roi_embeddings.shape
        num_candidates, cand_embed_dim = candidate_embeddings.shape
        if self.remove_duplicates:
            assert flattened_batch == num_candidates

        assert roi_embed_dim == cand_embed_dim
        similarity = (
            self.cosine_similarity(
                candidate_embeddings, roi_embeddings.unsqueeze(1)
            )
            / self.temperature
        )
        # print("similarity shape", similarity.shape, flattened_batch, num_candidates)
        assert similarity.shape == (
            flattened_batch,
            num_candidates,
        ), f"{similarity.shape} != ({flattened_batch}, {num_candidates})"

        if self.remove_duplicates:
            # If we have multiple positive pairs, we cannot use normal cross
            # entropy loss since it assumes only one positive class.
            # To get around this, we only keep the first instance of each
            # class.

            # First, let's create the overall similarity matrix.
            # ex: class_indices = [0, 1, 0, 2, 0]
            # out:
            # similarity_matrix = [
            #     [1, 0, 1, 0, 1],
            #     [0, 1, 0, 0, 0],
            #     [1, 0, 1, 0, 1],
            #     [0, 0, 0, 1, 0],
            #     [1, 0, 1, 0, 1],
            # ]
            similarity_matrix = class_indices.eq(
                class_indices[None, :].t()
            ).float()

            # We now identify which items in a row are duplicates. We do this by
            # first taking a cumulative sum of the similarity matrix
            # ex: Using the similarity_matrix above
            # out:
            # maybe_repeated_indices = [
            #     [1, 1, 2, 2, 3],
            #     [0, 1, 1, 1, 1],
            #     [1, 1, 2, 2, 3],
            #     [0, 0, 0, 1, 1],
            #     [1, 0, 2, 2, 3],
            # ]
            # Note that each time the number increases beyond 1, the position
            # where it increases is a duplicate.
            # We then subtract 1 to remove the first occurrence, and clamp between
            # 0 and 1 (treating all values > 1 as 1).
            # out:
            # maybe_repeated_indices = [
            #     [0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 1, 1, 1],
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 1, 1, 1],
            # ]
            maybe_repeated_indices = (
                similarity_matrix.cumsum(dim=1) - 1
            ).clamp(min=0, max=1)

            # Next, we need to simplify the above matrix to only include instances
            # that are both a maybe_duplicate and a positive class. That is,
            # keeping only the positions that are labels. We do this by doing an
            # element-wise multiplication between the similarity matrix and the
            # maybe_repeated_indices.
            # ex: Using the similarity_matrix and maybe_repeated_indices above
            # out:
            # unique_repeated_indices = [
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 0, 0, 0],
            #     [0, 0, 1, 0, 1],
            # ]
            unique_repeated_indices = torch.einsum(
                "ij, ij -> ij", similarity_matrix, maybe_repeated_indices
            )

            # Now that we have our beautiful matrix, we can now further simplify
            # to identify the columns that are duplicates. We do this by summing
            # along the rows, clamping between 0 and 1, and then tiling the result
            # to match the number of candidates.
            # We do this to create a mask that we can use to zero out the
            # influence the duplicate items for all examples, including for
            # negatives.
            # ex: Using the unique_repeated_indices above
            # out:
            # duplicate_mask = [
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 1, 0, 1],
            #     [0, 0, 1, 0, 1],
            # ]
            duplicate_mask = (
                (unique_repeated_indices.sum(dim=0))
                .clamp(max=1)
                .tile(num_candidates, 1)
            )

            # Now that we have the duplicate mask, we can now zero out the
            # similarity scores for the duplicate items. We set the similarity
            # of the duplicate items to -1e9 so that they dont contribute to the
            # loss. (softmax([0, 1, -1e9]) = [0.5, 0.5, 0] = softmax([0, 1])
            similarity = similarity.masked_fill(duplicate_mask.bool(), -1e9)
            similarity_matrix = similarity_matrix.masked_fill(
                duplicate_mask.bool(), 0
            )
            # and now we are done!
        else:
            # Since we are guaranteed to not have duplicate similarity scores,
            # we treat the similarity matrix as a one-hot matrix with the true
            # class indices.
            #valid_masks = y_true["valid_masks"]
            similarity_matrix = torch.zeros_like(similarity)
            similarity_matrix.scatter_(1, class_indices.unsqueeze(1), 1)

        # TODO(liamhebert): make sure the dimension is correct
        preds = torch.argmax(similarity, dim=1)
        valid_masks = y_true["valid_mask"]
        # print(similarity)
        loss = torch.nn.functional.cross_entropy(similarity, similarity_matrix)
        return (loss, preds, {"contrastive_loss": loss})


class SegmentationLoss(Loss):

    dice_loss: DiceLoss
    focal_loss: FocalLoss

    def __init__(self, weight_focal: float = 20.0, weight_dice: float = 1.0):
        """Compute the segmentation loss. This follows the loss described by
        the original SAM paper (https://arxiv.org/pdf/2304.02643), where loss
        is a linear combination of dice and focal loss. For DiceLoss, I follow
        the parameter settings used by MedSAM

        Args:
            weight_focal: the scaling factor for the focal loss. Defaults to 20
                per the original SAM paper.
            weight_dice: the scaling factor for the dice loss. Defaults to 1 per
                the original SAM paper.


        Returns:
            The mask segmentation loss.
        """
        super().__init__()
        self.dice_loss = DiceLoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )
        self.focal_loss = FocalLoss(use_softmax=False, reduction="mean")
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def __call__(
        self,
        pred_masks: torch.Tensor,
        gold_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the segmentation loss.

        Args:
            pred_masks: The masks predicted by SAM, with shape
                (batch_size, channel, height, width).
            gold_masks: the ground truth masks for each candidate term, with shape
                (batch_size, channel, height, width)
        Returns:
            The segmentation loss value.
        """
        #print("testing", pred_masks.shape, gold_masks.shape)
        dice = self.dice_loss(pred_masks, gold_masks)
        focal = self.focal_loss(pred_masks, gold_masks)
        loss = self.weight_focal * focal + self.weight_dice * dice
        return (loss, torch.empty(dice.shape), {"segmentation_loss": loss, "dice": dice, "focal": focal})


class CombinedLoss(Loss):

    contrastive_loss: ContrastiveLoss
    segmentation_loss: SegmentationLoss

    def __init__(
        self,
        weight_contrastive: float,
        weight_segmentation: float,
        contrastive_loss: ContrastiveLoss,
        segmentation_loss: SegmentationLoss,
    ):
        """Initializes the combined loss, which is just the sum of the
        contrastive and segmentation losses.

        Args:
            weight_contrastive (float): The weight for the contrastive loss.
            weight_segmentation (float): The weight for the segmentation loss.
            contrastive_loss (ContrastiveLoss): The contrastive loss. See
                ContrastiveLoss args for detailed description of parameters.
            segmentation_loss (SegmentationLoss): The segmentation loss. See
                SegmentationLoss args for detailed description of parameters.
        """

        super().__init__()
        self.weight_contrastive = weight_contrastive
        self.weight_segmentation = weight_segmentation
        self.contrastive_loss = contrastive_loss
        self.segmentation_loss = segmentation_loss
        self.remove_duplicates = True

    def __call__(
        self,
        roi_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        predicted_masks: torch.Tensor,
        y_true: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the combined segmentation and contrastive loss.

        Args:
            roi_embeddings: The embeddings of each region of interest, with
                shape (batch_size * num_rois, embedding_size).
            candidate_embeddings: The embeddings for each candidate item, with
                shape (num_candidates, embedding_size) if remove_duplicates is
                false, (batch_size * num_rois, embedding_size) otherwise.
            predicted_masks: The segmentation masks predicted by SAM, with shape
                (batch_size, channel, height, width)
            y_true: A dictionary containing the field "class_indices", which is a
                tensor of shape (batch_size * num_rois) containing the indices of
                the true class for each roi, and a field "gold_mask" containing
                the ground truth segmentation masks for each concept.

        Returns:
            The combined loss value.
        """
        if self.remove_duplicates:
            self.contrastive_loss.remove_duplicates = True
        else:
            self.contrastive_loss.remove_duplicates = False

        l1, preds, contrast_metrics = self.contrastive_loss(
            roi_embeddings, candidate_embeddings, y_true
        )
        gold_masks = y_true["gold_mask"]
        l2, _, seg_metrics = self.segmentation_loss(predicted_masks, gold_masks)
        #print("contrastive loss", l1, " segmentation metrics", l2, " full_loss", l1+l2)
        contrast_metrics.update(seg_metrics)
        return (
            self.weight_contrastive * l1 + self.weight_segmentation * l2,
            preds,
            contrast_metrics,
        )
