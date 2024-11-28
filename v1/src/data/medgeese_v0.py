"""
Dataset classes and utilities.
"""

from glob import glob
import json
import os
import os.path as osp

from data import medgeese_v0_utils as utils
from joblib import delayed
from joblib import Parallel
from lightning import LightningDataModule
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torchvision.tv_tensors as tv
from tqdm import tqdm
from transformers import AutoImageProcessor
from transformers import AutoTokenizer
from transformers import BatchEncoding
from utils import RankedLogger

tqdm.pandas()

logger = RankedLogger(__name__)

# TODO(liamhebert): Throughout this code, we only tokenize the images in the
# "get_item" method and we do not tokenize the text at all. We should instead
# tokenize ahead of time.


class MedGeeseDataModule(LightningDataModule):
    """DataModule containing processed train/val/test dataloaders for our
    dataset.

    This class handles the loading, splitting, and pre-processing of the dataset.

    Params:
        data_dir (str): The directory containing the raw dataset files.
        tensor_dir (str): The directory to save the processed tensors to.
        image_dir (str): The directory containing the images.
        train_batch_size (int): The total batch size for training. Must be
            divisible by the number of GPUs.
        test_batch_size (int): The total batch size for testing. Must be
            divisible by the number of GPUs.
        train_val_test_split (tuple[float, float, float]): A tuple containing the
            percentage split between train, val, and test datasets.
        num_workers (int): The number of workers to use for data loading.
        force_remake (bool): Whether to force remake the dataset cache. Relevant
            if the dataset is configured to cache to disk.
        pin_memory (bool): Whether to pin batches in GPU memory in the dataloader.
            This helps with performance on GPU, but can cause issues with large
            datasets.
        image_model_path (str): The path to the huggingface image model to use for
            tokenization.
        text_model_path (str): The path to the huggingface text model to use for
            tokenization.
        debug (bool): Whether to run in debug mode. In debug mode, only a subset
            of the dataset (first 25 rows) will be loaded. Default is False.
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
        text_model_path: str,
        debug: bool = False,
    ):
        assert (
            sum(train_val_test_split) == 1.0
        ), f"Train/val/test split must sum to 1.0. Got {train_val_test_split=}"
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        super().__init__()

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """

        data_dir = self.hparams.data_dir
        tensor_dir = self.hparams.tensor_dir
        image_dir = self.hparams.image_dir

        assert isinstance(data_dir, str)
        assert isinstance(tensor_dir, str)
        assert isinstance(image_dir, str)

        # TODO(liamhebert): Change osp.exists check to ensure we have exactly the
        # correct number of processed files, rather then just check if the
        # directory exists.
        if not self.hparams.force_remake and osp.exists(tensor_dir):
            logger.warning(
                f"Skipping data preparation"
                f"({not self.hparams.force_remake=} or {osp.exists(tensor_dir)=})"
            )
            return

        if self.hparams.force_remake and glob(tensor_dir + "/*.pt"):
            logger.warning(
                f"Removing existing tensor directory: {tensor_dir}"
                f"({self.hparams.force_remake=} and"
                f"{list(glob(tensor_dir + '/*.pt'))[:5]=})"
            )
            for file in tqdm(glob(tensor_dir + "/*.pt")):
                os.remove(file)

        logger.info(
            "Preparing data... "
            f"({self.hparams.force_remake=} or {not osp.exists(tensor_dir)=})"
        )

        # get umls master dict:
        with open(data_dir + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)

        text_tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.text_model_path
        )
        umls_text = [x["desc"] for x in umls_terms.values()]
        tokenized_umls = text_tokenizer(
            umls_text, return_tensors="pt", padding=True
        )
        assert isinstance(tokenized_umls, BatchEncoding)
        expanded_umls_values = [
            dict(zip(tokenized_umls.keys(), values))
            for values in zip(*tokenized_umls.values())
        ]

        for values, tokenized in zip(umls_terms.values(), expanded_umls_values):
            values["desc"] = tokenized
            values["idx"] = torch.tensor([values["idx"]])

        calc_mass_datasets = [
            "labels_calc_mammograms.csv",
            "labels_mass_mammograms.csv",
            "labels_calc_mammograms_test.csv",
            "labels_mass_mammograms_test.csv",
        ]
        calc_mass_dataframes = [data_dir + "/" + x for x in calc_mass_datasets]

        calc_mass_dataframe = utils.process_calc_mass_dataframes(
            calc_mass_dataframes
        )

        duke_dataframe = pd.read_csv(
            data_dir + "/duke_breast_cancer_annotations.csv"
        )
        duke_dataframe["organ"] = "Breast"
        # All examples in the duke dataset are breast tumor masses
        duke_dataframe["abnormality_type"] = "Breast Tumor Mass"

        liver_dataframe = utils.process_liver_dataset(
            data_dir + "/liver_dataset_fixed_trimmed.csv"
        )

        datasets = [calc_mass_dataframe, duke_dataframe, liver_dataframe]
        # datasets = [duke_dataframe]
        if self.hparams.debug:
            logger.warning(
                "Running in debug mode. Only loading first 25 rows of"
                "each dataset."
            )
            datasets = [x.head(25) for x in datasets]
        mega = pd.concat(datasets)

        # heres where the modifications begin
        mega["organ"] = mega["organ"].fillna("Breast")

        # just take the columns we need. if we want to keep all of them then
        # skip this part
        mega = mega[["image", "mask", "organ", "abnormality_type"]]

        mega["index"] = 1
        mega["index"] = mega["index"].cumsum() - 1  # 0, 1, 2, 3 etc.

        os.makedirs(tensor_dir, exist_ok=True)

        def process(row):
            if (
                os.path.exists(tensor_dir + "/" + str(row.index) + "-0.pt")
                and not self.hparams.force_remake
            ):
                return

            image_path = image_dir + "/" + str(row.image)
            mask_path = image_dir + "/" + str(row.mask)

            if not (os.path.exists(image_path) and os.path.exists(mask_path)):
                print("MISSING FILE: ", image_path, mask_path)
                return

            is_breast = row.organ == "Breast"

            try:
                img = (
                    Image.open(image_path)
                    if is_breast
                    else Image.fromarray(np.load(image_path))
                )
            except Exception as e:
                print("Error on image file when reading: " + image_path)
                print(e)
                return

            mask = (
                Image.open(mask_path)
                if is_breast
                else Image.fromarray(np.load(mask_path))
            )
            try:
                img = img.convert("RGB").resize((336, 336), Image.LANCZOS)

                mask = np.array(
                    mask.convert("RGB").resize((336, 336), Image.LANCZOS)
                )
            except Exception as e:
                print(
                    f"Error on file when resizing: {image_path} or {mask_path}"
                )
                print(e)
                return

            # separate masks into submasks
            # generate label to accompany each mask
            # get one for organ, one for abnormality, one for background
            masks = []
            umls_labels = []
            if is_breast:
                # we already have the tumor seg mask so just need breast +
                # background
                # if len(img.getbands()) == 1:
                #     img = img.convert("RGB")
                #     mask = mask.convert("RGB")
                breast = np.array(img)
                breast[breast > 0] = 255
                background = np.copy(breast)
                # invert 0s and 255s
                background ^= 255
                masks = [
                    Image.fromarray(background),
                    Image.fromarray(breast),
                    Image.fromarray(mask),
                ]
                umls_labels.append(umls_terms["Background"])
                umls_labels.append(umls_terms["Breast"])
                label = row.abnormality_type
                if label == "Breast Tumor Calc":
                    umls_labels.append(umls_terms["Calcinosis"])
                elif label == "Breast Tumor Mass":
                    umls_labels.append(umls_terms["Mass in breast"])
                else:
                    raise Exception("Label not recognized", label)
            else:
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
                        umls_terms[possible_values[int(kind)]]
                    )  # 0 is background, 1 is liver, 2 is tumor
                    masks.append(mask_kind)

            for i, (mask, label) in enumerate(zip(masks, umls_labels)):
                index = str(row.index)
                y = label["idx"]
                candidate_text = label["desc"]

                if np.max(np.array(mask)) == 0:
                    print("skipping empty mask")
                    continue

                torch.save(
                    (img, mask, y, candidate_text, row.organ),
                    (f"{tensor_dir}/{index}-{str(i)}.pt"),
                )

        logger.info("pre-tokenizing data....")
        Parallel(n_jobs=-1, backend="threading")(
            delayed(process)(row)
            for row in tqdm(mega.itertuples(index=False), total=len(mega))
        )
        # mega.progress_apply(process, axis=1)

    def setup(self, stage: str):
        """Load dataset for training/validation/testing.

        NOTE: When using DDP (multiple GPUs), this is run once per GPU.
        As a result, this function should be deterministic and not download
        or have side effects. As a result, all data processing should be done in
        prepare_data and cached to disk, or done prior to training.

        Args:
            stage: either 'fit' (train), 'validate', 'test', or 'predict'
        """
        logger.info(f"Setting up data for stage: {stage}")

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

        tensor_dir = self.hparams.tensor_dir  # type: ignore
        examples = list(glob(tensor_dir + "/*.pt"))
        train, val, test = self.hparams.train_val_test_split  # type: ignore
        train_set, val_test_set = train_test_split(
            examples, train_size=train, test_size=val + test
        )
        val_set, test_set = train_test_split(
            val_test_set, test_size=test / (val + test)
        )

        if self._train_dataset is None:
            # make training dataset
            self._train_dataset = MedGeeseDataset(
                items=train_set,
                model_path=self.hparams.image_model_path,  # type: ignore
            )
        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = MedGeeseDataset(
                items=val_set,
                model_path=self.hparams.image_model_path,  # type: ignore
            )
        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = MedGeeseDataset(
                items=test_set,
                model_path=self.hparams.image_model_path,  # type: ignore
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.train_batch_size,  # type: ignore
            shuffle=True,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.
        """
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.train_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the test dataloader.
        """
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
        items (list[str]): A list of paths to the processed tensors.
        image_model_name (str): The huggingface name of the image tokenizer to
        use.
    """

    def __init__(self, items: list[str], model_path: str):
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
        )

    def __getitem__(self, idx: int):
        """Fetch a single item from the dataset indexed by idx.

        Params:
            idx: The index of the item to fetch.

        Returns:
            A dictionary mapping keys to torch tensors. It is expected that the
            tensors have a shape of (batch_size, ...).
        """
        (img, mask, label, candidate_text, organ) = torch.load(self.items[idx])
        assert isinstance(img, Image.Image), f"{type(img)=}"
        assert isinstance(mask, Image.Image), f"{type(mask)=}"
        assert isinstance(label, torch.Tensor), f"{type(label)=}"
        assert isinstance(candidate_text, dict), f"{type(candidate_text)=}"
        assert all(isinstance(x, torch.Tensor) for x in candidate_text.values())
        assert isinstance(organ, str), f"{type(organ)=}"

        mask = tv.Mask(mask)
        img = tv.Image(img)
        if torch.max(mask) == 0:
            raise Exception("Empty mask pre")

        img, mask = self.safe_transforms(img, mask)
        try_img, try_mask = self.danger_transforms(img, mask)
        if torch.max(try_mask) != 0:
            img, mask = try_img, try_mask

        # This is where we tokenize the images
        # Because we do the random transforms as part of the __getitem__ method,
        # we need to tokenize the images here as well (and not ahead of time).
        img = self.processor(
            images=img,
            return_tensors="pt",
            do_normalize=True,
            do_rescale=True,
            do_center_crop=False,
            do_resize=True,
        )

        mask = mask.float()
        if torch.max(mask) == 0:
            raise Exception("Empty mask after")
        return {
            "x": {
                "candidate_input": candidate_text,
                "image_input": {"mask": mask, "img": img.pixel_values},
            },
            "y": label,
        }

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.items)
