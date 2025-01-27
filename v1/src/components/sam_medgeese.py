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
        self.text_model_path=text_model_path
        self.sequential = sequential
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.vision_model = AutoModel.from_pretrained('facebook/sam-vit-base')
        text_embedding_dim = self.text_model.config.hidden_size
        img_embedding_dim = self.vision_model.config.vision_config.hidden_size
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)
        self.vision_projection = nn.Linear(img_embedding_dim, projection_dim)
        self.processor = AutoProcessor.from_pretrained('facebook/sam-vit-base')
        '''
        self.sequential = sequential
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.text_model_path = text_model_path
        # TODO (carbonkat): make this path dynamic!
        self.vision_model = sam_model_registry["vit_b"](
            checkpoint=f"./segment_anything/checkpoints/{vision_model_path}"
        )

        text_embedding_dim = self.text_model.config.hidden_size
        image_embedding_dim = (
            self.vision_model.image_encoder.patch_embed.proj.out_channels
        )
        self.vision_norm = nn.LayerNorm(image_embedding_dim)

        self.patch_size = (
            self.vision_model.image_encoder.patch_embed.proj.kernel_size[0]
        )
        self.text_projection = nn.Linear(text_embedding_dim, projection_dim)
        if self.sequential:
            self.vision_projection = nn.Linear(
                image_embedding_dim, projection_dim
            )
        else:
            self.vision_projection = nn.Linear(
                self.vision_model.mask_decoder.transformer_dim, projection_dim
            )
            # self.vision_projection = nn.Linear()

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
        '''
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
        print(img.shape)
        #print(bounding_boxes.shape)
        if len(bounding_boxes.shape) < 3:
            if len(bounding_boxes.shape) == 1:
                bounding_boxes = bounding_boxes.unsqueeze(0)
            bounding_boxes = bounding_boxes[:, None, :]  # (B, 1, 4)
        if len(img.shape) < 4:
            img = img.unsqueeze(0)
        print(img.shape)
        sparse_prompt_embeddings, dense_prompt_embeddings = self.vision_model.prompt_encoder(input_points=None, input_labels=None, input_masks=None, input_boxes=bounding_boxes)
        image_embeddings = self.vision_model.vision_encoder(pixel_values=img)[0]
        image_positional_embeddings = self.vision_model.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.vision_model.mask_decoder.iou_token.weight, self.vision_model.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.vision_model.mask_decoder.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat_interleave(point_batch_size, 0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(point_batch_size, 0)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.vision_model.mask_decoder.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.vision_model.mask_decoder.num_mask_tokens), :]
        #roi_embeddings = mask_tokens_out.squeeze(1)
        #print("roi embeddings before project", roi_embeddings.shape)
        #roi_embeddings = self.vision_projection(roi_embeddings)
        #print("after projection", roi_embeddings.shape)
        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.vision_model.mask_decoder.upscale_conv1(image_embeddings)
        upscaled_embedding = self.vision_model.mask_decoder.activation(self.vision_model.mask_decoder.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.vision_model.mask_decoder.activation(self.vision_model.mask_decoder.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.vision_model.mask_decoder.num_mask_tokens):
            current_mlp = self.vision_model.mask_decoder.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.vision_model.mask_decoder.iou_prediction_head(iou_token_out)
        multimask_output = False
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]
        roi_embeddings = mask_tokens_out[:, :, mask_slice, :]
        #roi_embeddings = mask_tokens_out.squeeze(1)
        roi_embeddings = torch.sum(roi_embeddings, dim=1)
        roi_embeddings = roi_embeddings.squeeze(1)
        #roi_embeddings = self.vision_projection(roi_embeddings)
        #print("after projection", roi_embeddings.shape)
        candidate_embed = self.text_model(**candidate_input).pooler_output
        candidate_embed = self.text_projection(candidate_embed)

        masks = self.processor.post_process_masks(masks, original_sizes=[(1024, 1024)]*masks.shape[0], reshaped_input_sizes=[(1024, 1024)]*masks.shape[0], binarize=True, return_tensors='pt')
        masks = torch.stack(masks, dim=0)
        upscaled_masks = torch.sum(masks, dim=1)
        #print(masks.shape, masks.unique())
        '''
        #out = self.vision_model(pixel_values=img, input_boxes=bounding_boxes, output_hidden_states=True, multimask_output=False)
        #print(out)
        if len(bounding_boxes.shape) < 3:
        	bounding_boxes = bounding_boxes[:, None, :]  # (B, 1, 4)

        # The SAM image encoder includes a neck, which is used
        # to assist with mask decoding. For the embedding alignment portion,
        # we don't want this, just the standard VIT encoder part. I have modified
        #  the original codebase to return the last pre-neck embedding.
        img_embed, last_hidden_state = self.vision_model.image_encoder(img)
        # img_embed = (B, 64, 64, 256)
        # last_hidden_state = (B, 64, 64, image embedding dim (1280 for vit-h))

        # Obtain embeddings for bounding boxes/points (sparse embeddings) and
        # dense embeddings for masks.
        sparse_embeddings, dense_embeddings = self.vision_model.prompt_encoder(
            points=None,
            boxes=bounding_boxes,
            masks=None,
        )
        # sparse embeddings = (B, 2, 256)
        # dense_embeddings = (B, 256, 64, 64)

        candidate_embed = self.text_model(**candidate_input).pooler_output
        # In order to integrate text with the mask decoder, we need the embedding
        # to be in the same latent space. To accomplish this, first project, then
        # expand the tensor to be 3D in order to integrate it with the sparse/
        # dense embeddings. This needs to be done in order for the custom
        # attention mechanism to integrate text information.
        candidate_embed = self.text_projection(candidate_embed)  # (B, proj dim)

        pe = self.vision_model.prompt_encoder.get_dense_pe()
        pe = torch.repeat_interleave(pe, img.shape[0], dim=0)
        # This generates low resolution masks by decoding the image embedding,
        # sparse prompts, and dense prompts into masks. Right now, I only generate
        # one mask per image.
        low_res_masks, _, mask_embeddings = self.vision_model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            #text_embeddings=candidate_embed.unsqueeze(1),
            multimask_output=False,
        )
        # low_res_masks = (B, num_masks/batch, 256, 256)
        # mask_embeddings = (B, num_masks/batch, 256)

        # This converts the low resolution masks to the original image resolution.
        # Needed for IOU loss calculation
        upscaled_masks = self.vision_model.postprocess_masks(
            low_res_masks, (1024, 1024), (1024, 1024)
        )

        if self.sequential:
            # Threshold masks to produce binary outputs. Do we need to do this
            # with functionals?
            upscaled_binary_masks = (
                upscaled_masks > self.vision_model.mask_threshold
            )
            roi_embeddings = self.sequential_roi(
                upscaled_binary_masks, last_hidden_state
            )
        else:
            roi_embeddings = mask_embeddings.squeeze(1)
            roi_embeddings = self.vision_projection(roi_embeddings)
        '''
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
