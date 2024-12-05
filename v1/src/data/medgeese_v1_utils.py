"""
Various utility functions for processing the specific datasets that make up the
MedGeese dataset.
"""

import pandas as pd
import os
import numpy as np


def parse_file(dataset: str, path: str, candidates: list[str]) -> str:
    """
    Driver function for obtaining the correct UMLS mapping for nonuniform datasets

    Args:
        dataset (str): the name of the dataset
        path (str): the path to the file
        candidates (list[str]): a list of possible candidates for the given dataset

    Returns:
        str: a string pertaining to the correct matching
    """

    file = os.path.basename(path)
    if dataset == 'PAPILA':
        standard = file.lower()
        if 'disc' in standard:
            return candidates[0]
        return candidates[1]
    elif dataset == 'Breast-Ultrasound':
        standard = file.lower()
        if 'benign' in standard:
            return candidates[0]
        return candidates[1]
    
def expand_3d(image: np.array, mask: np.array) -> tuple[list[np.array], list[np.array]]:
    """
    Function for expanding 3D volumes and extracting nonzero-ed masks

    Args:
        image (np.array): numpy array of original 3D volume
        mask (np.array): numpy array of 3D volume masks

    Returns:
        tuple[list[np.array], list[np.array]]: paired list of expanded images and masks
    """

    images, masks = [], []
    for i in range(mask.shape[0]):
        if len(np.unique(mask[i])) == 1:
            continue
        else:  
            images.append(image[i])
            masks.append(mask[i])
    return (images, masks)
