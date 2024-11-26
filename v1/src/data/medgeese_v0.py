"""Dataset classes and utilities."""

from dataclasses import dataclass
import os.path as osp

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import numpy as np
import os
from PIL import Image
from glob import glob
import data.medgeese_v0_utils as utils
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import torchvision.tv_tensors as tv
from transformers import AutoImageProcessor

tqdm.pandas()


class MedGeeseDataModule(LightningDataModule):
    """DataModule containing processed train/val/test dataloaders for our
    dataset.

    This class handles the loading, splitting, and pre-processing of the dataset.

    Params:
        data_dir: The directory containing the raw dataset files
        train_batch_size: The total batch size for training. Must be divisible
            by the number of GPUs.
        test_batch_size: The total batch size for testing. Must be divisible by
            the number of GPUs.
        train_val_test_split: A tuple containing the percentage split between
            train, val, and test datasets.
        num_workers: The number of workers to use for data loading.
        force_remake: Whether to force remake the dataset cache. Relevant if the
            dataset is configured to cache to disk.
        pin_memory: Whether to pin batches in GPU memory in the dataloader. This
            helps with performance on GPU, but can cause issues with large
            datasets.
    """

    # Datasets are loaded in lazily during "setup" to assist with DDP
    _train_dataset: Dataset | None = None
    _val_dataset: Dataset | None = None
    _test_dataset: Dataset | None = None

    _train_device_batch_size: int = 1
    _test_device_batch_size: int = 1

    def __init__(
        self,
        data_dir: str,
        tensor_dir: str,
        image_dir: str,
        train_batch_size: int,
        test_batch_size: int,
        train_val_test_split: tuple[float, float, float],
        num_workers: int,
        force_remake: bool,
        pin_memory: bool,
        image_model_path: str,
    ):
        super().__init__()
        assert (
            sum(train_val_test_split) == 1.0
        ), f"Train/val/test split must sum to 1.0. Got {train_val_test_split=}"
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """
        # TODO(liamhebert): Implement data preparation logic
        if self.hparams.force_remake is False and osp.exists(self.tensor_dir):
            return

        # get umls master dict:
        # TODO: reorganize so it is easier to get specific terms
        with open(self.data_dir + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)

        calc_mass_datasets = [
            "labels_calc_mammograms.csv",
            "labels_mass_mammograms.csv",
            "labels_calc_mammograms_test.csv",
            "labels_mass_mammograms_test.csv",
        ]
        calc_mass_dataframes = [
            self.hparams.data_dir + "/" + x for x in calc_mass_datasets
        ]

        calc_mass_dataframe = utils.process_calc_mass_dataframes(
            calc_mass_dataframes
        )
        duke_dataframe = pd.read_csv(
            self.hparams.data_dir + "/duke_breast_cancer_annotations.csv"
        )
        liver_dataframe = utils.process_liver_dataset(
            self.hparams.data_dir + "/liver_dataset_fixed_trimmed.csv"
        )

        datasets = [calc_mass_dataframe, duke_dataframe, liver_dataframe]
        if self.hparams.debug:
            datasets = [x.head(25) for x in datasets]
        mega = pd.concat(datasets)

        # heres where the modifications begin
        mega["organ"] = mega["organ"].fillna("Breast")

        # just take the columns we need. if we want to keep all of them then
        # skip this part
        mega = mega[["train", "image", "mask", "organ", "abnormality_type"]]

        mega["index"] = 1
        mega["index"] = mega["index"].cumsum() - 1  # 0, 1, 2, 3 etc.

        os.makedirs(
            self.hparams.tensor_dir + "/processed_tensors", exist_ok=True
        )

        def process(row):
            if (
                os.path.exists(
                    self.hparams.tensor_dir
                    + "/processed_tensors/"
                    + str(row["index"])
                    + "-0.pt"
                )
                and self.force_remake == False
            ):
                return

            image_path = self.hparams.image_dir + "/" + str(row["image"])
            mask_path = self.hparams.image_dir + "/" + str(row["mask"])

            if (
                os.path.exists(image_path) == False
                or os.path.exists(mask_path) == False
            ):
                print("MISSING FILE: ", image_path, mask_path)
                return

            label = row["abnormality_type"]

            is_breast = row["organ"] == "Breast"
            try:
                img = (
                    Image.open(image_path)
                    if is_breast
                    else Image.fromarray(np.load(image_path))
                )
            except OSError:
                print("Error on file: " + image_path)
                return

            mask = (
                Image.open(mask_path)
                if is_breast
                else Image.fromarray(np.load(mask_path))
            )
            # separate masks into submasks
            # generate label to accompany each mask
            # get one for organ, one for abnormality, one for background
            masks = []
            umls_labels = []
            if is_breast:
                # we already have the tumor seg mask so just need breast + background
                if len(img.getbands()) == 1:
                    img = img.convert("RGB")
                    mask = mask.convert("RGB")
                breast = np.array(img)
                breast[breast > 0] = 255
                background = np.copy(breast)
                # invert 0s and 255s
                background ^= 255
                masks = [
                    Image.fromarray(background),
                    Image.fromarray(breast),
                    mask,
                ]
                umls_labels.append(umls_terms["Background"]["idx"])
                umls_labels.append(umls_terms["Breast"]["idx"])
                if label == "tumor_calc":
                    umls_labels.append(umls_terms["Calcinosis"]["idx"])
                elif label == "tumor_mass":
                    umls_labels.append(
                        umls_terms["Mass in breast"]["idx"]
                    )  # todo: ???????????
                else:
                    raise Exception("Label not recognized", label)
            else:
                # Liver dataset
                img = img.convert("RGB").resize(
                    (224, 224), Image.LANCZOS
                )  # only liver ones are not scaled
                mask = np.array(
                    mask.convert("RGB").resize((224, 224), Image.LANCZOS)
                )
                mask_arrays = [
                    np.copy(mask) for _ in range(len(np.unique(mask)))
                ]
                # make first mask only background -> we can def do this better
                # but just wanna make sure we get functionality 1st
                # Note: for background the white pixels will be the background
                # pixels, black are non-background
                # bground 1st, then liver, then tumor

                possible_values = [
                    "Background",
                    "Liver",
                    "Malignant neoplasm of liver",
                ]
                for mask_kind, kind in zip(mask_arrays, np.unique(mask)):
                    mask_kind[mask_kind == kind] = 255
                    mask_kind[mask_kind != 255] = 0
                    mask_kind = Image.fromarray(mask_kind)
                    umls_labels.append(
                        umls_terms[possible_values[int(kind)]]["idx"]
                    )  # 0 is background, 1 is liver, 2 is tumor
                    masks.append(mask_kind)

            labels = torch.tensor(umls_labels)

            for i, (m, y) in enumerate(zip(masks, labels)):
                index = str(row["index"])

                if np.max(np.array(m)) == 0:
                    print("skipping empty mask")
                    continue

                torch.save(
                    (img, m, y, row["organ"]),
                    f"{self.hparams.tensor_dir}/processed_tensors/{index}-{str(i)}.pt",
                )

        print("pre-tokenizing data....")
        mega.progress_apply(process, axis=1)

    def setup(self, stage: str):
        """Load dataset for training/validation/testing.

        NOTE: When using DDP (multiple GPUs), this is run once per GPU.
        As a result, this function should be deterministic and not download
        or have side effects. As a result, all data processing should be done in
        prepare_data and cached to disk, or done prior to training.

        Args:
            stage: either 'fit' (train), 'validate', 'test', or 'predict'
        """

        # We only have access to trainer in setup, so we need to calculate
        # these parameters here.
        if self.trainer is not None and (
            self._train_device_batch_size is None
            or self._test_device_batch_size is None
        ):
            # We test both here to fail quickly if misconfigured
            if (
                self.hparams.train_batch_size % self.trainer.world_size != 0
                or self.hparams.test_batch_size % self.trainer.world_size != 0
            ):
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible"
                    f"by the number of devices ({self.trainer.world_size})."
                )

            self._train_device_batch_size = (
                self.hparams.train_batch_size // self.trainer.world_size
            )
            self._test_device_batch_size = (
                self.hparams.test_batch_size // self.trainer.world_size
            )

        examples = list(
            glob(self.hparams.tensor_dir + "/processed_tensors/*.pt")  # type: ignore
        )
        train, val, test = self.hparams.train_val_test_split  # type: ignore
        train_set, val_test_set = train_test_split(
            examples, train_size=train, test_size=val + test
        )
        val_set, test_set = train_test_split(
            val_test_set, test_size=test / (val + test)
        )

        # TODO(liamhebert): Implement dataset setup logic
        # TODO(liamhebert): Implement debug mode
        if stage == "fit" and self._train_dataset is None:
            # make training dataset
            self._train_dataset = MedGeeseDataset(
                items=train_set, model_path=self.hparams.image_model_path  # type: ignore
            )
        elif stage == "validate" and self._val_dataset is None:
            # make validation dataset
            self._val_dataset = MedGeeseDataset(
                items=val_set, model_path=self.hparams.image_model_path  # type: ignore
            )
        elif (
            stage == "test" or stage == "predict"
        ) and self._test_dataset is None:
            # Make test dataset
            self._test_dataset = MedGeeseDataset(
                items=test_set, model_path=self.hparams.image_model_path  # type: ignore
            )
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.train_batch_size,  # type: ignore
            shuffle=True,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.train_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.test_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )


class MedGeeseDataset(Dataset):
    """Dataset instance for a dataloader.

    Params:
        df: The dataframe containing the dataset, used for tracking sizes.
        tensor_dir: The directory containing processed tensors.
    """

    def __init__(self, items: list, model_path: str):
        # assume our dataset contains image path, segmentation mask path, label
        self.items = items
        self.safe_transforms = v2.Compose(
            [
                v2.PILToTensor(),
                # v2.ToDtype(torch.float16),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )
        self.danger_transforms = v2.Compose([v2.RandomRotation(90)])
        self.processor = AutoImageProcessor.from_pretrained(
            model_path, local_files_only=False
        )  # set this to False if you want to download the tokenizer, you only need to do this once

    def __getitem__(self, idx: int):
        """Fetch a single item from the dataset indexed by idx.

        Params:
            idx: The index of the item to fetch.

        Returns:
            A dictionary mapping keys to torch tensors. It is expected that the
            tensors have a shape of (batch_size, ...).
        """
        img, mask, label, organ = torch.load(self.items[idx])
        mask = tv.Mask(mask)
        img = tv.Image(img)
        if torch.max(mask) == 0:
            raise Exception("Empty mask pre")

        img, mask = self.safe_transforms(img, mask)
        try_img, try_mask = self.danger_transforms(img, mask)
        if torch.max(try_mask) != 0:
            img, mask = try_img, try_mask
        # else:
        #     print('reverting danger transform, empty mask')

        # one_hots = encode_one_hot(umls_labels) # ??
        img = self.processor(
            images=img,
            return_tensors="pt",
            do_normalize=True,
            do_rescale=True,
            do_center_crop=False,
            do_resize=False,
        )
        # mask = self.processor(images=mask, return_tensors="pt", do_normalize=False, do_rescale=False, do_center_crop=False,  do_resize=False)
        # if len(labels) > 1 or len(labels) != len(mask):
        #     print(labels)
        # print(type(mask))
        mask = mask.float()
        if torch.max(mask) == 0:
            raise Exception("Empty mask after")
        return img, mask, label, organ

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.items)
