"""
Various utility functions for processing the specific datasets that make up the
VELCRO v1 dataset.
"""

import os

import numpy as np


def parse_concept_from_file_name(
    dataset: str, path: str, candidates: list[str]
) -> str:
    """
    Function for obtaining the correct UMLS mapping for datasets with
    terms embedded in the file structure.

    Args:
        dataset (str): the name of the dataset
        path (str): the path to the file
        candidates (list[str]): a list of possible candidates for the given
        dataset

    Returns:
        str: a string pertaining to the correct matching
    """

    # TODO(carbonkat): convert dataset concept directory from lists of
    # concepts to a dictionary of concepts.
    file = os.path.basename(path)
    standard = file.lower()
    if dataset == "PAPILA":
        if "disc" in standard:
            return candidates[0]
        return candidates[1]
    elif dataset == "Breast-Ultrasound":
        if "benign" in standard:
            return candidates[0]
        return candidates[1]
    elif dataset == "COVID-QU-Ex-lungMask_CovidInfection":
        if "non_covid" in standard:
            return candidates[0]
        elif "covid" in standard:
            return candidates[1]
    elif dataset == "COVID-19-Radiography-Database":
        if "normal" in standard:
            return candidates[0]
        elif "pneumonia" in standard:
            return candidates[1]
        elif "covid" in standard:
            return candidates[2]


def match_term_mask(
    masks: np.array, imgs: np.array, umls_terms: list[str]
) -> tuple[list[str], list[np.array]]:
    """
    Function to match pre-normalized masks in multi-object datasets to
    their corresponding UMLS terms.

    This function:
    1. Splits multi-object masks into single-object submasks
    2. Matches each submask to its corresponding UMLS term
    """

    filtered_masks = []
    candidates = []
    filtered_imgs = []
    for i, mask in enumerate(masks):
        # Should only be one unique label/mask now
        if np.unique(mask)[1] < len(umls_terms):
            candidate_label = umls_terms[np.unique(mask)[1]]
            candidates.append(candidate_label)
            filtered_masks.append(mask)
            filtered_imgs.append(imgs[i])

    return candidates, filtered_masks


def extract_2d_masks(
    image: np.array, mask: np.array
) -> tuple[list[np.array], list[np.array]]:
    """
    Function for expanding 3D volumes and extracting valid masks.
    Invalid masks are those which have no concept segmentation labels.
    Expects 3D volumes to be expanded along axis 0.

    Args:
        image (np.array): numpy array of original 3D volume
        mask (np.array): numpy array of 3D volume masks

    Returns:
        tuple[list[np.array], list[np.array]]: paired list of expanded
        images and masks
    """

    images, masks = [], []
    assert (
        len(image.shape) == 3 and len(mask.shape) == 3
    ), f"Unexpected image or mask shape. Expected (i, j, k), got (image) \
        {image.shape=} and (mask) {mask.shape=}."

    for i in range(mask.shape[0]):
        if len(np.unique(mask[i])) == 1 and np.unique(mask[i])[0] == 0:
            print("gothere!")
            continue
        else:
            images.append(image[i])
            masks.append(mask[i])
    return (images, masks)


def multi_mask_processing(
    images: np.array, masks: np.array, dataset: str
) -> tuple[list[np.array], list[np.array]]:
    """
    Function for mask normalization and submask expansion. For single-concept
    masks, normalizes all masked components to 255. For multi-concept
    masks, splits into single-concept submasks but does not normalize (to
    allow for UMLS lookup later)

    Args:
        images (np.array): list of 2D image arrays.
        masks (np.array): list of single or multi-concept 2D mask arrays.

    Returns:
        tuple[list[np.array], list[np.array]]: paired list of expanded
        images and masks.
    """

    expanded_masks = []

    # List of datasets which require additional processing.
    multi_label_datasets = [
        "crossmoda",
        "SpineMR",
        "AMOS",
        "AMOSMR",
        "AbdomenCT1K",
        "COVID-19-20-CT-SegChallenge",
        "COVID19-CT-Seg-Bench",
        "CT_AbdTumor",
        "TCIA-LCTSC",
        "CT-ORG",
        "TotalSeg_cardiac",
        "TotalSeg_muscles",
        "TotalSeg_organs",
    ]

    # TODO(carbonkat): add multi_label_dataset as flag in dataframe.
    if dataset not in multi_label_datasets:
        for mask in masks:
            #mask[mask != 0] = 255
            expanded_masks.append(mask)
        return images, expanded_masks
    else:
        new_images = []
        for mask, img in zip(masks, images):
            # Generate a new submask which only contains
            # labels for a single concept in multi-concept
            # datasets.
            submasks = np.unique(mask)[1:]
            for label in submasks:
                new_submask = mask.copy()
                # Zero out all pixels related to other concepts,
                # but do not standardize to 255. This is because
                # we need to use the labels to match to their
                # corresponding UMLS terms later on.
                new_submask[mask != label] = 0
                expanded_masks.append(new_submask)
                new_images.append(img.copy())

        return new_images, expanded_masks
