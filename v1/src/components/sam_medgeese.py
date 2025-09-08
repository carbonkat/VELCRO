"""
Code related to VELCRO v2.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel, AutoProcessor, SamImageProcessor

class SAMVELCRO(TwoTowerEncoder):
    """
    Model that generates masks and matches patch embeddings
    to text embeddings using SAM.
    """

    text_model: AutoModel
    vision_model: AutoModel
    patch_size: int = 16

    def __init__(
        self,
        text_model_path: str = "bert-base-uncased",
        vision_model_path: str = "facebook/sam-vit-base",
        projection_dim: int = 512,
        sequential: bool = True,
        freeze_text_enc: bool = False,
        freeze_img_enc: bool = False,
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
            freeze_text_enc (bool): whether to freeze the text encoder
            freeze_img_enc (bool): whether to freeze the image encoder
        """
        super().__init__()
        self.text_model_path = text_model_path
        self.sequential = sequential

        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.vision_model = AutoModel.from_pretrained(vision_model_path)

        text_embedding_dim = self.text_model.config.hidden_size
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)

        # SAMImageProcessor is used to post-process masks into the original image size
        self.processor = SamImageProcessor.from_pretrained(vision_model_path)

        # Freeze encoders if specified
        if freeze_img_enc:
            for name, layer in self.vision_model.named_children():
                if name in ['vision_encoder']:
                    for param in layer.parameters():
                        param.requires_grad = False
        if freeze_text_enc:
            for name, layer in self.text_model.named_children():
                if name in ['encoder']:
                    for param in layer.parameters():
                        param.requires_grad = False

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
        img = img.squeeze(1)

        # Prepare embeddings to be passed into the SAM mask decoder
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
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]

        # Concatenate output tokens (IoU tokens and mask embedding tokens)
        output_tokens = torch.cat(
            [
                self.vision_model.mask_decoder.iou_token.weight,
                self.vision_model.mask_decoder.mask_tokens.weight,
            ],
            dim=0,
        )
        # Since the model expects one bounding box per image, we need to
        # ensure that the bounding box prompt embeddings are in the correct shape.
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
        # Retrieve the mask and IoU tokens from the decoder output.
        # Ignore the bounding box token, which is the first token.
        mask_tokens_out = point_embedding[
            :, :, 1:(1 + self.vision_model.mask_decoder.num_mask_tokens), :
        ]
        # Reshape and upscale image embeddings to be in the correct shape 
        # for the hypernetwork prediction of masks.
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

        # Use the mask tokens and the upscaled embeddings to predict masks
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
        # Select the correct mask or masks for output. In our case,
        # we are only interested in the first mask.
        mask_slice = slice(0, 1)

        masks = masks[:, :, mask_slice, :, :]
        # Get the mask embedding of the first mask. This is our contextual
        # mention/RoI embedding.
        roi_embeddings = mask_tokens_out[:, :, 0, :]

        roi_embeddings = roi_embeddings.squeeze(1)

        assert len(roi_embeddings.shape) == 2, roi_embeddings.shape

        # Embed the candidate text using the BERT text encoder
        candidate_embed = self.text_model(**candidate_input).pooler_output
        candidate_embed = self.text_projection(candidate_embed)

        # Post-process the masks to be in the original image size. This
        # requires the original image size, which we assume to be 1024x1024.
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
