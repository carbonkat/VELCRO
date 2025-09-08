"""
Code related to VELCRO CLIP baselines.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel
from transformers import CLIPVisionModel
from transformers import ViTModel


class ClipFull(TwoTowerEncoder):
    """
    CLIP baseline. This model can be used to perform both whole-image
    and mention-level image-text matching.
    """

    # TODO(liamhebert): Do we want to use a ClipTextModelWithProjection here?
    text_model: AutoModel
    vision_model: ViTModel
    patch_size: int = 16

    def __init__(
        self,
        text_model_path: str = "bert-base-uncased",
        vision_model_path: str = "openai/clip-vit-large-patch14",
        is_clip: bool = True,
        projection_dim: int = 512,
    ):
        """Constructs the model.

        Args:
            text_model_path (str): The huggingface model identifier for the text
                model.
            vision_model_path (str): The huggingface model identifier for the
                vision model.
            is_clip (bool, unused): Whether the vision model is a CLIP model or not.
            projection_dim (int, unused): The dimension to project the embeddings to.
        """
        super().__init__()
        self.full_model = AutoModel.from_pretrained(vision_model_path)

        self.text_model_path = text_model_path

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
        img = image_input["img"]
        img = img.squeeze(1)
        outs = self.full_model(pixel_values=img, **candidate_input)
        img_embeddings = outs.image_embeds
        text_embeddings = outs.text_embeds
        
        return img_embeddings, text_embeddings
