import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    AutoModelForMaskedLM,
    AutoModelForImageClassification,
)

from model import TwoTowerEncoder
from torch import Tensor


class ClipMedGeese(TwoTowerEncoder):
    # TODO(liamhebert): Do we want to use a ClipTextModelWithProjection here?
    text_model: AutoModelForMaskedLM
    vision_model: AutoModelForImageClassification
    patch_size: int = 16

    def __init__(
        self, text_model_path: str, vision_model_path: str, patch_size: int
    ):
        super().__init__()
        self.text_model = AutoModelForMaskedLM.from_pretrained(text_model_path)
        self.vision_model = AutoModelForImageClassification.from_pretrained(
            vision_model_path
        )
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
