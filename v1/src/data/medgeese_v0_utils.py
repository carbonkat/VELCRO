import pandas as pd


def process_calc_mass_dataframes(paths: list[str]) -> pd.DataFrame:
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
    return pd.concat(final_dfs)


def process_liver_dataset(path: str) -> pd.DataFrame:
    # liver dataset
    liver_dataset = pd.read_csv(path)
    liver_dataset.dropna(subset="cancer", inplace=True)
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
