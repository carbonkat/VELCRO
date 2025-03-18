"""
Code related to Medgeese v2.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel
from segment_anything import sam_model_registry
from transformers import AutoProcessor
import gc

class SAMMedGeese(TwoTowerEncoder):
    """
    Model that generates masks and matches patch embeddings
    to text embeddings using SAM.
    """

    text_model: AutoModel
    vision_model: sam_model_registry  # type: ignore
    patch_size: int = 16

    def __init__(
        self,
        text_model_path: str = "bert-base-uncased",
        vision_model_path: str = "sam_vit_b.pth",
        projection_dim: int = 512,
        sequential: bool = True,
    ):
        """Constructs the model.

        Args:
            text_model_path (str): The huggingface model identifier for the text
                model.
            vision_model_path (str): The SAM checkpoint identifier for the
                vision model. This should be located in the segment_anything
                folder
            projection_dim (int): The dimension of the shared embedding space
                which the text and ROI embeddings will be projected into.
            sequential (bool): whether to generate ROI embeddings sequentially (in
                the same style as V1-CLIP), or using native mask embeddings (SAM)
        """
        super().__init__()
        self.text_model_path = text_model_path
        self.sequential = sequential
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.vision_model = AutoModel.from_pretrained("facebook/sam-vit-base")
        text_embedding_dim = self.text_model.config.hidden_size
        img_embedding_dim = self.vision_model.config.vision_config.hidden_size
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)
        #self.vision_projection = nn.Linear(img_embedding_dim, projection_dim)
        self.processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

    def forward(
        self,
        candidate_input: dict[str, Tensor],
        image_input: dict[str, Tensor],
        bounding_boxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Generates the required embeddings for the text and image inputs.

        Args:
            candidate_input (dict): Dict of the inputs required for the candidate
                model.
            image_input (dict): Dict of the inputs required for the image tower.
            bounding_boxes (torch.Tensor): Tensors for bounding boxes.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Returns three
            tensors, the first representing the candidate embeddings, the second
            representing the ROI embeddings, and the third representing the
            computed SAM mask(s).
        """
        img = image_input["img"]
        # NOTE: this is bad practice and for some reason only occurs in the test set; should definitely hunt down the cause of this!
        if len(img.shape) > 4:
            img = img.squeeze()
        sparse_prompt_embeddings, dense_prompt_embeddings = (
            self.vision_model.prompt_encoder(
                input_points=None,
                input_labels=None,
                input_masks=None,
                input_boxes=bounding_boxes,
            )
        )
        image_embeddings: torch.Tensor = self.vision_model.vision_encoder(pixel_values=img)[0]
        image_positional_embeddings = (
            self.vision_model.get_image_wide_positional_embeddings()
        )
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat(
            [
                self.vision_model.mask_decoder.iou_token.weight,
                self.vision_model.mask_decoder.mask_tokens.weight,
            ],
            dim=0,
        )
        assert point_batch_size == 1, point_batch_size
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        point_embeddings = tokens.to(
            self.vision_model.mask_decoder.iou_token.weight.dtype
        )

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, _ = (
            self.vision_model.mask_decoder.transformer(
                point_embeddings=point_embeddings,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                attention_similarity=None,
                target_embedding=None,
                output_attentions=None,
            )
        )

        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[
            :, :, 1:(1 + self.vision_model.mask_decoder.num_mask_tokens), :
        ]

        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.vision_model.mask_decoder.upscale_conv1(
            image_embeddings
        )
        upscaled_embedding = self.vision_model.mask_decoder.activation(
            self.vision_model.mask_decoder.upscale_layer_norm(
                upscaled_embedding
            )
        )
        upscaled_embedding = self.vision_model.mask_decoder.activation(
            self.vision_model.mask_decoder.upscale_conv2(upscaled_embedding)
        )

        hyper_in_list = []
        for i in range(self.vision_model.mask_decoder.num_mask_tokens):
            current_mlp = (
                self.vision_model.mask_decoder.output_hypernetworks_mlps[i]
            )
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(
            batch_size, point_batch_size, num_channels, height * width
        )
        masks = (hyper_in @ upscaled_embedding).reshape(
            batch_size, point_batch_size, -1, height, width
        )

        multimask_output = False
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, :, mask_slice, :, :]

        roi_embeddings = mask_tokens_out[:, :, 0, :]

        roi_embeddings = roi_embeddings.squeeze(1)
        assert len(roi_embeddings.shape) == 2, roi_embeddings.shape

        candidate_embed = self.text_model(**candidate_input).pooler_output
        candidate_embed = self.text_projection(candidate_embed)

        masks = self.processor.post_process_masks(
            masks,
            original_sizes=[(1024, 1024)] * masks.shape[0],
            reshaped_input_sizes=[(1024, 1024)] * masks.shape[0],
            binarize=False,
            return_tensors="pt",
        )
        upscaled_masks = torch.stack(masks, dim=0).squeeze(1)

        return (
            roi_embeddings,
            candidate_embed,
            upscaled_masks,
        )

    def sequential_roi(
        self, masks: torch.Tensor, last_hidden_state: torch.Tensor
    ):
        """
        Perform ROI embedding sequentially in the same manner as v1-CLIP.

        Args:
            masks (torch.Tensor): the upscaled masks produced by the
                mask decoder after post-processing and binarization.
                Expected shape is (B, C, H, W)
            last_hidden_state (torch.Tensor): the image embedding retrieved from
                the ViT image encoder before layer normalization.
                Expected shape is (B, 64, 64, image embedding dim)

        Returns:
            torch.Tensor: normalized region of interest embeddings.
        """
        # Take the upscaled masks and apply convolution to project
        # into hidden image embedding space.
        masks = (self.expand_mask_kernel(masks) > 0).float()
        masks = masks.flatten(start_dim=1).float()

        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)
        img_embed = self.vision_norm(last_hidden_state)

        # Project into desired shared image-text space.
        projected_img_embed = self.vision_projection(img_embed)
        mask_embeds = torch.einsum("bij, bi -> bj", projected_img_embed, masks)
        mask_embeds = mask_embeds.squeeze(-1)
        # Since the dot product above sums the tokens that make up the mask, we
        # now have to normalize the mask embeddings by the number of tokens that
        # make up the mask (ie: taking the mean). This is because some masks can
        # be larger then others.

        # First, we calculate the number of mask tokens per image. This is just
        # done by summing the mask over the last dimension.
        mask_size = masks.sum(dim=-1, keepdim=True)
        # Then we divide the mask embeddings by the mask size to get the average
        normalized_mask_embeds = mask_embeds / mask_size

        return normalized_mask_embeds
