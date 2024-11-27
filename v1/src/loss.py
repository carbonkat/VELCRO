"""
Loss classes.
"""

from abc import ABC
from abc import abstractmethod

import torch
from torch import nn


class Loss(ABC):
    """
    Abstract class for loss functions.
    """

    @abstractmethod
    def __call__(
        self,
        roi_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        y_true: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    cosine_similarity: nn.CosineSimilarity = torch.nn.CosineSimilarity(dim=1)
    remove_duplicates: bool = True

    def __init__(self, remove_duplicates: bool = True):
        """Initializes the contrastive loss.

        Args:
            remove_duplicates (bool, optional): Whether to remove duplicate values
            from the loss calculation. If you are guaranteed to not have duplicate
            candidates, then this should be set to False for a speed up. Defaults
            to True.
        """
        self.remove_duplicates = remove_duplicates
        super().__init__()

    def __call__(
        self,
        roi_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        y_true: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        class_indices = y_true["class_indices"]
        flattened_batch, roi_embed_dim = roi_embeddings.shape
        num_candidates, cand_embed_dim = candidate_embeddings.shape
        if not self.remove_duplicates:
            assert flattened_batch == num_candidates

        assert roi_embed_dim == cand_embed_dim

        # NOTE: This has multiple positives for each negative, which is not
        # typical for a classification task, and may break cross entropy loss.
        similarity_matrix = class_indices.eq(class_indices[None, :].t()).float()
        similarity = self.cosine_similarity(
            roi_embeddings, candidate_embeddings
        )

        if self.remove_duplicates:
            # To get around this, we can keep only the first instance of a
            # positive class and negative pair.
            maybe_repeated_indices = (
                similarity_matrix.cumsum(dim=1) - 1
            ).clamp(min=0, max=1)
            unique_repeated_indices = torch.einsum(
                "ij, ij -> ij", similarity_matrix, maybe_repeated_indices
            )
            duplicate_mask = (
                (unique_repeated_indices.sum(dim=0) - 1)
                .clamp(min=0, max=1)
                .tile(num_candidates, 1)
            )

            # We set the similarity of the duplicate items to -1e9 so that they
            # dont contribute to the loss.
            similarity = similarity.masked_fill(duplicate_mask.bool(), -1e9)
            similarity_matrix = similarity_matrix.masked_fill(
                duplicate_mask.bool(), 0
            )

        # TODO(liamhebert): make sure the dimension is correct
        preds = torch.argmax(similarity, dim=1)
        return (
            torch.nn.functional.cross_entropy(similarity, similarity_matrix),
            preds,
        )
