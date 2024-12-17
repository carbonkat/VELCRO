"""
A really quick script for generating SAM masks for v1 dataset images.

TODO(carbonkat): Move the segment anything stuff + this script
into its own folder.
"""

import os

import numpy as np
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from skimage.measure import regionprops
from tqdm import tqdm
import data.medgeese_v1_utils as utils


def main():

    # TODO(carbonkat): link this with the datadir parameter set in the paths file
    # in configs (possibly use https://click.palletsprojects.com/en/stable/).
    data_dir = "<data_dir here>"
    mask_dir = "<mask_dir here>"

    # TODO(carbonkat): make this path dynamic
    sam = sam_model_registry["default"](checkpoint="<sam_checkpoint_path here>")
    sam.to(device="cuda")
    mask_predictor = SamPredictor(sam)

    file_folder = []
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    # This walks through the data directory structure, retrieves
    # the desired files for processing, gets their parent folders
    # (modality folder, i.e. XRay and subfolder i.e. specific dataset
    # they originally belonged to), and makes a mask directory
    # with this same structure to put the new data into.
    for root, _, files in os.walk(data_dir):
        if mask_dir in root:
            continue
        else:
            for file in files:
                if file.endswith(".npz"):
                    original_path = root
                    modality_subfolder = original_path.replace(data_dir, "")[1:]
                    new_mask_subfolder = os.path.join(
                        mask_dir, modality_subfolder
                    )
                    # Make new mask directories if they do not already exist
                    os.makedirs(new_mask_subfolder, exist_ok=True)

                    full_path = os.path.join(root, file)
                    file_folder.append(full_path)

    for f in tqdm(file_folder):
        new_folder = os.path.dirname(f).replace(data_dir, "")[1:]
        mask_folder = os.path.join(mask_dir, new_folder)

        packed_data = np.load(f)
        img = packed_data["imgs"]
        mask = packed_data["gts"]

        if len(np.unique(mask)) == 1:
            continue

        data_dict = {"imgs": img}

        # If a 3D volume, get image and mask slices for
        # each timestep.
        if len(img.shape) > 2 and img.shape[2] != 3:
            imgs, masks = utils.extract_2d_masks(img, mask)
        else:
            imgs = [img]
            masks = [mask]

        all_masks = []
        for mask, candidate_img in zip(imgs, masks):
            # SAM expects RBG images, so need to convert to correct
            # format.
            if len(candidate_img.shape) < 3:
                candidate_img = np.asarray(
                    Image.fromarray(candidate_img).convert("RGB")
                )

            mask_predictor.set_image(candidate_img)
            # Compute bounding boxes for each segmentation artifact
            # in the original mask.
            bboxes = regionprops(mask)
            new_masks = []
            for prop in bboxes:
                bbox = prop.bbox
                prop_x = bbox[1]
                prop_y = bbox[0]
                prop_x2 = bbox[3]
                prop_y2 = bbox[2]

                # Generate a SAM mask for each bounding box. The boxes are
                # provided to guide generation in order to produce good-faith
                # segmentations.
                SAM_masks, _, _ = mask_predictor.predict(
                    box=np.array([prop_x, prop_y, prop_x2, prop_y2]),
                    multimask_output=False,
                )
                m = SAM_masks.squeeze()
                new_masks.append(m)

            blank_canvas = np.zeros_like(mask)

            # Iterate through each mask and gradually add to
            # the full reconstructed mask. NOTE: this assumes
            # all segmentations belong to the same concept.
            for m in new_masks:
                blank_canvas[m != 0] = 255
            all_masks.append(blank_canvas)

        final_mask = np.asarray(all_masks).squeeze()
        assert final_mask.shape == mask.shape
        data_dict["gts"] = final_mask
        np.savez(
            os.path.join(mask_folder, os.path.basename(f)),
            imgs=data_dict["imgs"],
            gts=data_dict["gts"],
        )


if __name__ == "__main__":
    main()
