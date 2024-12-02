"""
Code related to Medgeese v0.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel
from transformers import CLIPVisionModel
from transformers import ViTModel


class ClipMedGeese(TwoTowerEncoder):
    """
    Model that matches patch embeddings to text embeddings, similar to CLIP.
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
            patch_size (int): The cnn patch size used for tokenization. This is
                used to expand the pixel-level mask to the correct image patches.
                This number can often be retrieved by looking at the model's
                name. For example: "openai/clip-vit-large-patch14" has a patch
                size of 14.
        """
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_path)
        if is_clip:
            self.vision_model = CLIPVisionModel.from_pretrained(
                vision_model_path
            )
            # Weirdly, the CLIP model does not use the layer norm if you access
            # the hidden states directly. To allow for gradient watching, we have
            # to access the layer norm directly to apply it.
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L1104-L1116
            self.vision_layer_norm = (
                self.vision_model.vision_model.post_layernorm
            )
        else:
            self.vision_model = ViTModel.from_pretrained(vision_model_path)
            self.vision_layer_norm = nn.Identity()

        self.text_model
        self.vision_model

        text_embedding_dim = self.text_model.config.hidden_size
        vision_embedding_dim = self.vision_model.config.hidden_size

        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)
        self.vision_projection = nn.Linear(vision_embedding_dim, projection_dim)

        self.patch_size = self.vision_model.config.patch_size

        # Because masks are given as a region within pixel space (255, 255),
        # we need to map them to the tokens that the vision model creates.
        # To do this, we tokenize the masks similarly to how the images are
        # tokenized, just without the learned CNN weights.
        #
        # In practice, this is done by defining a convolutional kernel that has
        # the same kernel size (re: patch size) and stride as the underlying
        # vision model. Then, we set the weights of the kernel to be all ones so
        # that if any pixel in the stride is masked, the resulting patch is
        # positive. We then treat all positive patches, now with the same number
        # of tokens as the vision model, as the desired areas of interest.

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
        # TODO(liamhebert): Ideally, we should have "img" be a field and we map
        # it to self.vision model (ie: self.vision_model(**image_input["img"]))
        # That way it can be flexible in case other models have different input
        # types.
        img = image_input["img"]
        img = img.squeeze(1)
        img_embed = self.vision_model(pixel_values=img)

        # First token is the CLS embedding, which we do not want.
        img_embed = self.vision_layer_norm(
            img_embed.last_hidden_state[:, 1:, :]
        )
        projected_img_embed = self.vision_projection(img_embed)

        mask = image_input["mask"]
        mask = (self.expand_mask_kernel(mask) > 0).float()
        mask = mask.flatten(start_dim=1).float()

        # If I understand correctly, this mask is made up of ones and zeroes. As
        # a result, if you multiply it by the projected_embeddings, you would
        # zero out the embeddings that are not in the mask.
        # If that is correct, then this will do element-wise multiplication to
        # zero out the embeddings that are not in the mask
        mask_embeds = torch.einsum("bij, bi -> bj", projected_img_embed, mask)
        mask_embeds = mask_embeds.squeeze(-1)

        # TODO(liamhebert): Explain what this is doing
        mask_size = mask.squeeze(-1).sum(dim=-1).unsqueeze(-1)

        normalized_mask_embeds = mask_embeds / mask_size

        candidate_embed = self.text_model(**candidate_input).pooler_output
        candidate_embed = self.text_projection(candidate_embed)

        return candidate_embed, normalized_mask_embeds
