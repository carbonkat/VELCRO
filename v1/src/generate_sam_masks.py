"""
A really quick script for generating SAM masks for v1 dataset images.
"""

import os

from numpy import load
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from skimage.measure import regionprops
from tqdm import tqdm


# TODO(carbonkat): Link this to the function in medgeese_v1_utils.py which does
# the same thing. I think this will require treating the dataset_files.v1 folder
# as a package.
def expand_3d(
    image: np.array, mask: np.array
) -> tuple[list[np.array], list[np.array]]:
    """
    Function for expanding 3D volumes and extracting nonzero-ed masks.
    Expects 3D volumes to be expanded along axis 0.

    Args:
        image (np.array): numpy array of original 3D volume
        mask (np.array): numpy array of 3D volume masks

    Returns:
        tuple[list[np.array], list[np.array]]: paired list of expanded
        images and masks
    """

    images, masks = [], []
    for i in range(mask.shape[0]):
        if len(np.unique(mask[i])) == 1:
            continue
        else:
            images.append(image[i])
            masks.append(mask[i])
    return (images, masks)


def main():
    """Main function to generate SAM masks for v1 dataset images."""

    # TODO(carbonkat): link this with the datadir parameter set in the paths file
    # in configs.
    data_dir = "<insert data directory here>"
    mask_dir = "<insert desired mask directory here>"

    # TODO(carbonkat): make this path dynamic
    sam = sam_model_registry["default"](
        checkpoint="<insert checkpoint file path here>"
    )
    sam.to(device="cuda")
    mask_predictor = SamPredictor(sam)

    file_folder = []
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)

    for root, _, files in os.walk(data_dir):
        for file in files:

            # TODO(carbonkat): make this much more elegant
            if file.endswith(".npz") and mask_dir not in root:
                original_path = root
                modality_subfolder = original_path.replace(data_dir, "")[1:]
                modality_folder = os.path.dirname(modality_subfolder)

                new_mask_folder = os.path.join(mask_dir, modality_folder)
                new_mask_subfolder = os.path.join(mask_dir, modality_subfolder)
                # Make new mask directories if they do not already exist
                if not os.path.exists(new_mask_folder):
                    os.mkdir(new_mask_folder)
                if not os.path.exists(new_mask_subfolder):
                    os.mkdir(new_mask_subfolder)

                full_path = os.path.join(root, file)
                # Skip over files if they have already been generated
                if os.path.exists(os.path.join(new_mask_subfolder, file)):
                    continue
                else:
                    file_folder.append(full_path)

    for f in tqdm(file_folder):
        new_folder = os.path.dirname(f).replace(data_dir, "")[1:]
        mask_folder = os.path.join(mask_dir, new_folder)

        packed_data = load(f)
        try:
            img = packed_data["imgs"]
            mask = packed_data["gts"]
        except Exception as e:
            print(packed_data.keys())
            print(e)

        if len(np.unique(mask)) == 1:
            continue

        data_dict = {"imgs": img}

        # If a 3D volume, get image and mask slices for
        # each timestep.
        if len(img.shape) > 2 and img.shape[2] != 3:
            imgs, masks = expand_3d(img, mask)
        else:
            imgs = [img]
            masks = [mask]

        all_masks = []
        for i, candidate_img in enumerate(imgs):
            # SAM expects RBG images, so need to convert to correct
            # format.
            if len(candidate_img.shape) < 3:
                candidate_img = np.asarray(
                    Image.fromarray(candidate_img).convert("RGB")
                )

            mask_predictor.set_image(candidate_img)
            # Compute bounding boxes for each segmentation artifact
            # in the original mask.
            bboxes = regionprops(masks[i])
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

            blank_canvas = np.zeros_like(masks[i])

            # Iterate through each mask and gradually add to
            # the full reconstructed mask. NOTE: this assumes
            # all segmentations belong to the same concept.
            for m in new_masks:
                blank_canvas[m != 0] = 255
            all_masks.append(blank_canvas)

        if len(all_masks) > 1:
            final_mask = np.asarray(all_masks)
        else:
            final_mask = all_masks[0]

        data_dict["gts"] = final_mask
        np.savez(
            os.path.join(mask_folder, os.path.basename(f)),
            imgs=data_dict["imgs"],
            gts=data_dict["gts"],
        )


if __name__ == "__main__":
    main()
