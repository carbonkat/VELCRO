"""
Various utility functions for processing the specific datasets that make up the
MedGeese dataset.
"""

import pandas as pd
import os


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