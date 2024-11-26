"""
Code related to Medgeese v0.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModelForImageClassification
from transformers import AutoModelForSequenceClassification


class ClipMedGeese(TwoTowerEncoder):
    """
    Model that matches patch embeddings to text embeddings, similar to CLIP.
    """

    # TODO(liamhebert): Do we want to use a ClipTextModelWithProjection here?
    text_model: AutoModelForSequenceClassification
    vision_model: AutoModelForImageClassification
    patch_size: int = 16

    def __init__(
        self,
        text_model_path: str = "bert-base-uncased",
        vision_model_path: str = "openai/clip-vit-large-patch14",
        patch_size: int = 14,
    ):
        """Constructs the model.

        Args:
            text_model_path (str): The huggingface model identifier for the text
                model.
            vision_model_path (str): The huggingface model identifier for the
                vision model.
            patch_size (int): The cnn patch size used for tokenization. This is
                used to expand the pixel-level mask to the correct image patches.
                This number can often be retrieved by looking at the model's
                name. For example: "openai/clip-vit-large-patch14" has a patch
                size of 14.
        """
        super().__init__()
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_model_path
        )
        self.vision_model = AutoModelForImageClassification.from_pretrained(
            vision_model_path
        )
        # TODO(liamhebert): We can probably grab this directly from the
        # vision_model object, rather then relying on the user to pass it in.
        self.patch_size = patch_size

        self.expand_mask_kernel = nn.Conv2d(
            3,
            1,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=False,
        )
        self.expand_mask_kernel.requires_grad_(False)
        self.expand_mask_kernel.weight = nn.Parameter(
            torch.ones_like(self.expand_mask_kernel.weight), requires_grad=False
        )

    def forward(
        self, candidate_input: dict[str, Tensor], image_input: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Generates the required embeddings for the text and image inputs.

        Args:
            candidate_input (dict): Dict of the inputs required for the candidate
                model.
            image_input (dict): Dict of the inputs required for the image tower.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Returns two tensors, the first
            representing the candidate embeddings and the second representing the
            image embeddings.
        """
        img = image_input["pixel_values"]
        img = img.squeeze(1)
        img_embed = self.vision_model(pixel_values=img)

        mask = image_input["mask"]
        mask = (self.expand_mask_kernel(mask) > 0).float()
        mask = mask.flatten(start_dim=1).unsqueeze(-1).float()

        # TODO(liamhebert): Explain wtf is happening here (ie: why transpose)
        mask_embeds = torch.matmul(img_embed.transpose(-2, -1), mask)
        mask_embeds = mask_embeds.squeeze(-1)

        # TODO(liamhebert): Explain what this is doing
        mask_size = mask.squeeze(-1).sum(dim=-1).unsqueeze(-1)

        normalized_mask_embeds = mask_embeds / mask_size

        candidate_embed = self.text_model(**candidate_input).pooler_output

        return candidate_embed, normalized_mask_embeds
