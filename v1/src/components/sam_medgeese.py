"""
Code related to Medgeese v2.
"""

from model import TwoTowerEncoder
import torch
from torch import nn
from torch import Tensor
from transformers import AutoModel
from segment_anything import sam_model_registry


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
    ):
        """Constructs the model.

        Args:
            text_model_path (str): The huggingface model identifier for the text
                model.
            vision_model_path (str): The SAM checkpoint identifier for the
                vision model. This should be located in the segment_anything
                folder
            patch_size (int): The cnn patch size used for tokenization. This is
                used to expand the pixel-level mask to the correct image patches.
                Since SAM's vision encoder uses VIT, this can be retrieved from
                accessing the vision encoder's parameters.
            is_clip (bool): A flag indicating whether or not a CLIP text model
                should be used for text encoding. This would take the place of
                the custom text prompt encoder. (Will likely be removed later)
        """
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_path)
        # TODO (carbonkat): make this path dynamic!
        self.vision_model = sam_model_registry["vit_b"](
            checkpoint=f"/home/carbok/MedGeese/v1/src/segment_anything/{vision_model_path}"
        )
        self.vision_layer_norm = nn.Identity()

        self.text_model
        self.vision_model

        text_embedding_dim = self.text_model.config.hidden_size
        image_embedding_dim = (
            self.vision_model.image_encoder.patch_embed.proj.out_channels
        )
        self.patch_size = (
            self.vision_model.image_encoder.patch_embed.proj.kernel_size[0]
        )
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)
        self.vision_projection = nn.Linear(image_embedding_dim, projection_dim)

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
            1,
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
        # TODO(liamhebert): Ideally, we should have "img" be a field and we map
        # it to self.vision model (ie: self.vision_model(**image_input["img"]))
        # That way it can be flexible in case other models have different input
        # types.
        img = image_input["img"]
        if len(bounding_boxes.shape) == 2:
            bounding_boxes = bounding_boxes[:, None, :]  # (B, 1, 4)

        # The SAM image encoder includes a neck, which is used
        # to assist with mask decoding. For the embedding alignment portion,
        # we don't want this, just the standard VIT encoder part. I have modified
        #  the original codebase to return the last pre-neck embedding. I
        # hope this has a similar effect to what the CLIP version does.
        img_embed, last_hidden_state = self.vision_model.image_encoder(img)
        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)

        # Obtain embeddings for bounding boxes/points (sparse embeddings) and
        # dense embeddings for masks. The original paper chooses to treat
        # "text" inputs as sparse embeddings (though they don't actually train
        # on text).
        sparse_embeddings, dense_embeddings = self.vision_model.prompt_encoder(
            points=None,
            boxes=bounding_boxes,
            masks=None,
        )

        # This gets the [CLS] token
        candidate_embed = self.text_model(**candidate_input).pooler_output
        # In order to integrate text with the mask decoder, we need
        # the embedding to be in the same latent space. To accomplish
        # this, first project, then expand the tensor to be 3D in order
        # to integrate it with the sparse/dense embeddings. This needs
        # to be done in order for the custom attention mechanism to
        # integrate text information. The text can be either expanded
        # to be sparse or dense. For now, I am making it sparse since this
        # requires less shape manipulation, but eventually we should test
        # both.
        candidate_embed = self.text_projection(candidate_embed)
        text_decoder_embed = candidate_embed.unsqueeze(1).expand(
            -1, sparse_embeddings.shape[1], -1
        )

        # This generates low resolution masks by decoding the
        # image embedding, sparse prompts, and dense prompts
        # into masks. Right now, I only generate one mask per image.
        # I believe the low_res_masks are logits, which is needed for
        # loss calculation.
        low_res_masks, _ = self.vision_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.vision_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            text_embeddings=text_decoder_embed,
            multimask_output=False,
        )

        # TODO(carbonkat): make the resizing dynamic!
        # This converts the low resolution masks to the original
        # image resolution. Needed for IOU loss calculation
        # NOTE: SAM expects images to be of size 1024x1024,
        # which is very memory intensive. I would like to
        # make the images smaller without significantly changing
        # the architecture, if possible, but this appears quite
        # difficult to do.
        upscaled_masks = self.vision_model.postprocess_masks(
            low_res_masks, (1024, 1024), (1024, 1024)
        )

        # Take the upscaled masks and apply convolution to project
        # into hidden image embedding space.
        masks = (self.expand_mask_kernel(upscaled_masks) > 0).float()
        masks = masks.flatten(start_dim=1).float()

        img_embed = self.vision_layer_norm(last_hidden_state)

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

        return (
            normalized_mask_embeds,
            candidate_embed,
            upscaled_masks,
            image_input["mask"],
        )
