"""
Dataset classes and utilities for the V2 SAM data. The V2 dataset is composed of
image, segmentation mask pairs pertaining to medically relevant image artifacts
found in various modalities (CT, mammograms, ultrasounds, etc). Medical artifacts
(organs, clinically significant findings) are linked to their corresponding UMLS
terms stored in a separate directory. This is very similar to the v1 data, except
that here we are also generating bounding boxes as prompts for SAM.
"""

from glob import glob
import json
import os
import os.path as osp

from data import medgeese_v1_utils as m_utils
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
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import BatchEncoding
from utils import RankedLogger
from skimage.measure import regionprops
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

tqdm.pandas()

logger = RankedLogger(__name__)

# TODO(liamhebert): Throughout this code, we only tokenize the images in the
# "get_item" method. We should instead tokenize ahead of time.

def collate_fn(data):
        new_candidate_input = dict()
        for key in data[0]['x']['candidate_input']:
            new_candidate_input[key] = torch.stack([d['x']['candidate_input'][key] for d in data], dim=0)

        new_image_input = dict()
        for key in data[0]['x']['image_input']:
            new_image_input[key] = torch.squeeze(torch.stack([d['x']['image_input'][key] for d in data], dim=0))

        bounding_boxes = [torch.permute(d['x']['bounding_boxes'], (1, 2, 0)) for d in data]
        bounding_boxes = torch.squeeze(pad_sequence(bounding_boxes, batch_first=True))
        class_indices = torch.stack(([d['y']['class_indices'] for d in data]), dim=0)

        return {
            "x": {
                "candidate_input": new_candidate_input,
                "image_input": new_image_input,
                "bounding_boxes": bounding_boxes
            },
            "y": {"class_indices": class_indices},
        }

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
            Currently not implemented, so will raise an error if True.
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
        tokenizer_path: str,
        debug: bool = False,
    ):
        assert (
            sum(train_val_test_split) == 1.0
        ), f"Train/val/test split must sum to 1.0. Got {train_val_test_split=}"

        if debug:
            raise Exception(
                "Feature not implemented. Please switch debug mode to False."
            )

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        super().__init__()

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """
        print("Starting v2 processing!")
        data_dir = os.path.join(self.hparams.data_dir, "v1")

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
        umls_path = data_dir
        with open(umls_path + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)

        # Tokenize the UMLS terms
        text_tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.text_model_path
        )
        # We tokenize all the terms together so that we don't have to worry about
        # padding issues when we batch the data. That is, it will automatically
        # pad all the terms to be the same length (the largest sequence).
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
            values["idx"] = torch.tensor(values["idx"])

        master_files = []
        folders = []

        # TODO(carbonkat): implement logic to isolate multi-concept
        # masks and split into single-concept masks. This may require
        # pulling the original datasets and performing manual preprocessing.
        # For now, all multi-concept datasets have been removed from the
        # v1 dataset directory.
        img_mask_path = os.path.join(data_dir, 'ground_truths')
        for root, _, files in os.walk(img_mask_path):
            for file in files:
                if file.endswith(".npz"):
                    folders.append(os.path.basename(root))
                    master_files.append(os.path.join(root, file))

        master_files = master_files[0:25]
        folders = folders[0:25]
        mega = pd.DataFrame({"File": master_files, "Dataset": folders})

        mega["index"] = 1
        mega["index"] = mega["index"].cumsum() - 1  # 0, 1, 2, 3 etc.

        # A dictionary to assist with mapping UMLS terms to dataset instances.
        # Mapping is performed on a per-dataset basis to make adding new datasets
        # easier.
        with open(umls_path + "/" + "dataset_directory.json") as json_file:
            term_mapping = json.load(json_file)

        os.makedirs(tensor_dir, exist_ok=True)

        # Function for resizing and processing masks to convert them into tensors.
        def process(row):
            if (
                os.path.exists(tensor_dir + "/" + str(row.index) + "-0.pt")
                and not self.hparams.force_remake
            ):
                return

            index = str(row.index)
            dataset = row.Dataset
            packed_data = np.load(row.File)
            img = packed_data["imgs"]
            mask = packed_data["gts"]

            if len(np.unique(mask)) == 1:
                return

            # TODO(kathryncarbone): add test to make sure the mask and
            # image shape are the same

            if len(img.shape) > 2 and img.shape[2] != 3:
                # Make sure that 3D volumes have the same shape between images and
                # masks. If not, then there is no 1-1 matching between image and
                # mask slices.
                assert (
                    img.shape == mask.shape
                ), f"3D volume shapes do not match. Got (image) \
                    {img.shape=} and (mask) {mask.shape=}."
                # This converts 3D volumes into 2D slices, with each image slice
                # corresponding to a mask slice
                imgs, masks = m_utils.extract_2d_masks(img, mask)
            else:
                # It is possible for images to be RGB and masks to
                # be greyscale/2D arrays. To check shape agreement,
                #only check the first and second shapes
                assert (
                    img.shape[0] == mask.shape[0] and
                    img.shape[1] == mask.shape[1]
                ), f"Image and mask shapes do not match. Got (image) \
                    {img.shape=} and (mask) {mask.shape=}."
                imgs = [img]
                masks = [mask]

            # For multi-concept datasets, split up masks so that each submask
            # only contains the segmentation labels of a single concept.
            imgs, masks = m_utils.multi_mask_processing(imgs, masks, dataset)

            potential_terms = term_mapping[dataset]
            # Grabbing UMLS terms and standardizing list length between
            # images, masks, and terms.
            if len(potential_terms) == 1:
                # If this file is from a one-concept dataset, no need
                # for additional parsing. The list of candidate terms
                # must still be extended to the length of the image list
                # to account for 3D volumes, though.
                candidate_terms = [umls_terms[potential_terms[0]]] * len(imgs)

            elif np.max(masks[0]) == 255:
                # Extract appropriate concept from dataset files where
                # the correct concept is embedded in the file name.
                # In this case, masks are already standardized but the
                # length of the potential terms is greater than 1. This
                # is different from multi-concept datasets, where pixel
                # values represent different classes and are thus not
                # normalized.
                candidate_terms = [
                    umls_terms[
                        m_utils.parse_concept_from_file_name(
                            dataset, row.File, potential_terms
                        )
                    ]
                ] * len(imgs)
                if candidate_terms[0] is None:
                    return
            else:
                # Otherwise, match appropriate term to mask label for all masks.
                candidate_mini_terms, masks = m_utils.match_term_mask(
                    masks, imgs, potential_terms
                )
                candidate_terms = [
                    umls_terms[term] for term in candidate_mini_terms
                ]

            for i, (img, mask, term) in enumerate(
                zip(imgs, masks, candidate_terms)
            ):
                
                y = term["idx"]
                candidate_text = term["desc"]
                try:

                    img = (
                        Image.fromarray(img)
                        .convert("RGB")
                    #    .resize((1024, 1024), Image.LANCZOS)
                    )
                    mask = (
                        Image.fromarray(mask)
                        .convert("RGB")
                    #    .resize((1024, 1024), Image.LANCZOS)
                    )

                    x_scale = 1024 / mask.size[0]
                    y_scale = 1024 / mask.size[1]

                    bboxes = regionprops(np.asarray(mask))
                    reordered_boxes = []
                    for prop in bboxes:
                        bbox = prop.bbox
                        # Convert coordinates from y1,x1,y2,x2 to x1,y1,x2,y2
                        x1 = int(np.round(bbox[1] * x_scale))
                        y1 = int(np.round(bbox[0] * y_scale))
                        xmax = int(np.round(bbox[3] * x_scale))
                        ymax = int(np.round(bbox[2] * y_scale))
                        reordered_boxes.append([x1, y1, xmax, ymax])
                        #reordered_boxes.append([bbox[1], bbox[0], bbox[3], bbox[2]])
                    img = img.resize((1024, 1024), Image.LANCZOS)
                    mask = mask.resize((1024, 1024), Image.LANCZOS)
                    print(img.size, mask.size)
                    #print("number of bounding boxes:", len(reordered_boxes))
                    # TODO(liamhebert): Ensure that files are saved in a somewhat
                    # standardized way to match the rest of the datasets. For
                    # instance, datasets in v1 are saved as npz files with
                    # dictionaries, rather than a tuple.
                    torch.save(
                        (img, mask, y, candidate_text, reordered_boxes),
                        (f"{tensor_dir}/{index}-{i}.pt"),
                    )

                except Exception as e:
                    print(f"Error on file when resizing: {row.File}")
                    print(img.shape, mask.shape)
                    print(e)

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
                model_path=self.hparams.tokenizer_path,  # type: ignore
                is_testing=False,
            )
        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = MedGeeseDataset(
                items=val_set,
                model_path=self.hparams.tokenizer_path,  # type: ignore
                is_testing=False,
            )
        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = MedGeeseDataset(
                items=test_set,
                model_path=self.hparams.tokenizer_path,  # type: ignore
                is_testing=True,
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
            collate_fn=collate_fn, # type: ignore
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
            collate_fn=collate_fn # type: ignore
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

    def __init__(self, items: list[str], model_path: str, is_testing: bool):
        # assume our dataset contains image path, segmentation mask path, label
        self.items = items
        self.is_testing = is_testing
        self.safe_transforms = v2.Compose(
            [
                v2.PILToTensor(),
                # v2.ToDtype(torch.float16),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )
        self.danger_transforms = v2.Compose([v2.RandomRotation(90)])
        #self.processor = segment_anything.sam_model_registry['default'](checkpoint=
        #    f'/home/carbok/MedGeese/v1/src/segment_anything/{model_path}')
        
        self.processor = AutoProcessor.from_pretrained(
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
        (img, mask, label, candidate_text, bboxes) = torch.load(self.items[idx])
        assert isinstance(img, Image.Image), f"{type(img)=}"
        assert isinstance(mask, Image.Image), f"{type(mask)=}"
        assert isinstance(label, torch.Tensor), f"{type(label)=}"
        assert isinstance(candidate_text, dict), f"{type(candidate_text)=}"
        assert all(isinstance(x, torch.Tensor) for x in candidate_text.values())
        assert isinstance(bboxes, list), f"{type(bboxes)=}"

        mask = tv.Mask(mask)
        img = tv.Image(img)
        if torch.max(mask) == 0:
            raise Exception("Empty mask pre")

        #TODO(carbonkat): if we do transforms, we need to also change the bounding
        # boxes to match the new orientation. Alternately, we could just generate
        # the bounding boxes here, though this might be more expensive

        if not self.is_testing:
            img, mask = self.safe_transforms(img, mask)
            try_img, try_mask = self.danger_transforms(img, mask)
            if torch.max(try_mask) != 0:
                img, mask = try_img, try_mask

        # This is where we tokenize the images
        # Because we do the random transforms as part of the __getitem__ method,
        # we need to tokenize the images here as well (and not ahead of time).
        #img = self.processor.preprocess(img)
        inputs = self.processor(
            images=img,
            input_boxes = [bboxes],
            return_tensors="pt",
            do_normalize=True,
            do_rescale=True,
            #do_center_crop=False,
            do_resize=True,
        )

        print(inputs)

        mask = mask.float()
        if torch.max(mask) == 0:
            raise Exception("Empty mask after")
        return {
            "x": {
                "candidate_input": candidate_text,
                "image_input": {"mask": mask, "img": inputs.pixel_values},
                "bounding_boxes": inputs.input_boxes
            },
            "y": {"class_indices": label},
        }

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.items)
