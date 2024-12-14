"""
A really quick script for generating SAM masks for v1 dataset images.

TODO(carbonkat): Move the segment anything stuff + this script
into its own folder.
"""

import os

import numpy as np
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from skimage.measure import regionprops
from tqdm import tqdm
import data.medgeese_v1_utils as utils


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

    # TODO(carbonkat): link this with the datadir parameter set in the paths file
    # in configs (possibly use https://click.palletsprojects.com/en/stable/).
    data_dir = "/home/carbok/MedGeese/v1/src/data/dataset_files/v1/ground_truths"
    mask_dir = "/home/carbok/MedGeese/v1/src/data/dataset_files/v1/masks"

    # TODO(carbonkat): make this path dynamic
    sam = sam_model_registry["default"](
        checkpoint="/home/carbok/MedGeese/v1/src/segment_anything/sam_vit_h_4b8939.pth"
    )
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
                    #if os.path.exists(os.path.join(new_mask_subfolder, file)):
                    #    continue
                    #else:
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
            imgs, masks = utils.expand_2d(img, mask)
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
