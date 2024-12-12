"""
Various utility functions for processing the specific datasets that make up the
MedGeese v0 dataset.
"""

import pandas as pd


def process_calc_mass_dataframes(paths: list[str]) -> pd.DataFrame:
    """Processes the calc and mass dataframes into a single dataframe.

    Notably, this function will:
    - Standardize the names of the columns
    - Drop the columns we don't need
    - Add a column for the organ (Breast)
    - Standardize the abnormality_type to be either "Breast Tumor Mass" or
      "Breast Tumor Calc", based on the path.

    NOTE: This function assumes that the paths are named in a way that indicates
    whether the data is for a mass or calc dataset (looking for those keywords).

    Args:
        paths (list[str]): The paths to the calc and mass dataframes.

    Raises:
        ValueError: If the path is invalid (missing either "mass" or "calc"),
            this function will raise a ValueError.

    Returns:
        pd.DataFrame: One standardized dataframe merging together the calc and
            mass dataframes.
    """
    final_dfs: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "image file path": "image",
                "cropped image file path": "cropped",
                "ROI mask file path": "mask",
                "abnormality type": "abnormality_type",
            }
        )
        # if malignant mass detected, change to tumor
        df = df[
            ["image", "mask", "abnormality_type", "pathology"]
        ]  # keep just what we need
        # we dont care if it's malignant or benign, just if it's a tumor or
        # not
        df["organ"] = "Breast"

        if "mass" in path:
            df["abnormality_type"] = "Breast Tumor Mass"
        elif "calc" in path:
            df["abnormality_type"] = "Breast Tumor Calc"
        else:
            raise ValueError(
                f'Unexpected path: {path}. Expecting path with "mass" or "calc"'
                "in name."
            )
        final_dfs.append(df)
    return pd.concat(final_dfs)


def process_liver_dataset(path: str) -> pd.DataFrame:
    """Processes the liver dataframe into a standardized format.

    Notably, this function will:
    - Rename columns to be standardized
    - Replace integer values with human-readable strings, for both the organ
        and abnormality_type columns.

    Args:
        path (str): Path to the liver dataset.

    Returns:
        pd.DataFrame: The standardized liver dataset.
    """
    # liver dataset
    liver_dataset = pd.read_csv(path)
    liver_dataset.rename(
        columns={
            "liver": "organ",
            "cancer": "abnormality_type",
            "slice_id": "patient_id",
        },
        inplace=True,
    )
    liver_dataset.replace(
        {
            "organ": {0: "Background", 1: "Liver"},
            "abnormality_type": {0: "None", 1: "Liver Tumor"},
        },
        inplace=True,
    )
    return liver_dataset
