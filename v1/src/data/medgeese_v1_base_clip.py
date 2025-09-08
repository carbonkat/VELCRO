"""
Dataset classes and utilities for the V2 data. The V2 dataset is composed of
image, segmentation mask pairs pertaining to medically relevant image artifacts
found in various modalities (CT, mammograms, ultrasounds, etc). Medical artifacts
(organs, clinically significant findings) are linked to their corresponding UMLS
terms stored in a separate directory.
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
from torchvision.transforms import PILToTensor, ToPILImage
import torchvision.tv_tensors as tv
from tqdm import tqdm
from transformers import AutoImageProcessor
from transformers import AutoTokenizer
from transformers import CLIPTokenizer
from transformers import BatchEncoding
from utils import RankedLogger
from itertools import groupby
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DistributedSampler
import random
from catalyst.data.sampler import DistributedSamplerWrapper
from skimage.measure import regionprops

tqdm.pandas()

logger = RankedLogger(__name__)

class VELCRODataModule(LightningDataModule):
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
        debug: bool = False,
        data_source: str = "ground_truths",
        crop: bool = False,
        from_mem: bool = False,
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

        data_dir = os.path.join(self.hparams.data_dir, "v1")

        tensor_dir = self.hparams.tensor_dir
        image_dir = self.hparams.image_dir

        assert isinstance(data_dir, str)
        assert isinstance(tensor_dir, str)
        assert isinstance(image_dir, str)

        # TODO: Change osp.exists check to ensure we have exactly the
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

        if "openai" in self.hparams.text_model_path:
            text_tokenizer = CLIPTokenizer.from_pretrained(
                self.hparams.text_model_path
            )
        else:
            # Tokenize the UMLS terms
            text_tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.text_model_path
            )
        print(text_tokenizer)
        # We tokenize all the terms together so that we don't have to worry about
        # padding issues when we batch the data. That is, it will automatically
        # pad all the terms to be the same length (the largest sequence).
        umls_text = [x["desc"] for x in umls_terms.values()]
        tokenized_umls = text_tokenizer(
            umls_text, return_tensors="pt", padding=True, truncation=True
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
        img_mask_path = os.path.join(data_dir, self.hparams.data_source)
        for root, _, files in os.walk(img_mask_path):
            for file in files:
                if file.endswith(".npz"):
                    folders.append(os.path.basename(root))
                    master_files.append(os.path.join(root, file))

        mega = pd.DataFrame({"File": master_files, "Dataset": folders})
        print(mega["Dataset"].value_counts())

        mega["index"] = 1
        mega["index"] = mega["index"].cumsum() - 1  # 0, 1, 2, 3 etc.

        # A dictionary to assist with mapping UMLS terms to dataset instances.
        # Mapping is performed on a per-dataset basis to make adding new datasets
        # easier.
        with open(umls_path + "/" + "dataset_directory.json") as json_file:
            term_mapping = json.load(json_file)

        os.makedirs(tensor_dir, exist_ok=True)
        os.makedirs(tensor_dir + "/masks", exist_ok=True)
        os.makedirs(tensor_dir + "/annotations", exist_ok=True)
        os.makedirs(tensor_dir + "/images", exist_ok=True)

        # Optional variables to track the incidence of invalid masks and images
        self.bad_masks = 0
        self.bad_imgs = 0

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
            if len(np.unique(mask)) == 1 and np.unique(mask)[0] == 0:
                print(row.File, np.unique(mask))
                return

            # TODO(carbonkat): add test to make sure the mask and
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
                if len(masks) != mask.shape[0]:
                    print(row.File)
                    # throw_count+=1
            else:
                # It is possible for images to be RGB and masks to
                # be greyscale/2D arrays. To check shape agreement,
                # only check the first and second shapes
                assert (
                    img.shape[0] == mask.shape[0]
                    and img.shape[1] == mask.shape[1]
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

            else:
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

            # Iterate through all image, mask, and term triplets
            for i, (img, mask, term) in enumerate(
                zip(imgs, masks, candidate_terms)
            ):

                y = term["idx"]
                candidate_text = term["desc"]
                mask = mask.astype(np.uint8)

                try:
                    # First get all unique labels in the mask. If there is more than one,
                    # we need to create separate masks for each label.
                    label_ids = np.unique(mask)[1:]
                    bad_masks_count = 0
                    mask = Image.fromarray(mask)

                    # Iterate through all unique labels (mention instances) in the mask
                    for k, label in enumerate(label_ids):
                        segment_mask = np.zeros_like(np.asarray(mask))
                        segment_mask[np.asarray(mask) == label] = 1
                        segment_mask = Image.fromarray(segment_mask)
                        assert segment_mask.size == mask.size, segment_mask.size
                        
                        # Resize to 224x224 for ViT input. Also resize to 1024x1024 to ensure that
                        # masks are valid for both SAM and CLIP (this prevents issues where some
                        # masks are valid at 224x224 but become empty when resized to 1024x1024 and
                        # vice versa).
                        resized = segment_mask.resize((224, 224), Image.NEAREST).convert("RGB")
                        resized = PILToTensor()(resized)
                        resized = resized.numpy()
                        check = segment_mask.resize((1024, 1024), Image.NEAREST).convert("RGB")
                        check = PILToTensor()(check)
                        check = check.numpy()
                        
                        # Enumerate counters for bad masks. An image is only bad if all
                        # masks related to it are bad.
                        if len(np.unique(check)) == 1 or len(np.unique(resized)) == 1:
                            self.bad_masks += 1
                            bad_masks_count += 1
                            continue
                        
                        # Crop masks and images to bounding box of the visual mention if specified
                        if self.hparams.crop:
                            new_mask = np.asarray(mask)
                            segment_mask = np.zeros_like(new_mask)
                            segment_mask[new_mask == label] = 1
                            assert segment_mask.shape == new_mask.shape, segment_mask.shape

                            y_indices, x_indices = np.where(segment_mask > 0)
                            x_min, x_max = np.min(x_indices), np.max(x_indices)
                            y_min, y_max = np.min(y_indices), np.max(y_indices)
                            cropped_img = img[
                                y_min:y_max,
                                x_min:x_max,
                            ]
                            
                            # After cropping, resize the image back to 224x224 for ViT input
                            cropped_img = Image.fromarray(cropped_img).convert("RGB")
                            cropped_img = cropped_img.resize((224, 224), Image.LANCZOS)
                            cropped_img.save(
                                (f"{tensor_dir}/masks/{index}-{i}-{k}-{y}.png"),
                            )
                            torch.save(
                                (y, candidate_text),
                                (f"{tensor_dir}/annotations/{index}-{i}-{k}-{y}.pt")
                            )
                    
                    img = Image.fromarray(img).convert("RGB")
                    img = img.resize((224, 224), Image.LANCZOS)

                    # If all masks for this image were bad, then the image is bad
                    if bad_masks_count == len(label_ids):
                        self.bad_imgs+=1
                        continue
                    else:
                        img.save(
                            (f"{tensor_dir}/images/{index}-{i}-{y}.png"),
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
        self._train_device_batch_size = self.hparams.train_batch_size // self.trainer.world_size
        self._test_device_batch_size = self.hparams.test_batch_size // self.trainer.world_size

        if self.trainer is not None and (
            self._train_device_batch_size is None
            or self._test_device_batch_size is None
        ):
            print("trainer got here!")
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
        examples = sorted(list(glob(tensor_dir + "/masks/*.png")))
        random.seed(42)

        # Get classes of all examples for dataset balancing
        all_check = []
        for i in examples:
            count = int(os.path.basename(i).split("-")[-1].split(".")[0])
            all_check.append(count)

        # This is similar to MedSAM's data processing approach, where they randomly select
        # one bounding box from the set of possible bounding boxes for a single datapoint.
        # The actual dataset size is only the amount of distinct images in the dataset
        new_dataset_size = int(len(list(glob(tensor_dir + "/images/*.png"))))

        # Group datapoints by case to ensure no data leakage happens
        by_case = [
            list(i)
            for j, i in groupby(
                examples, lambda x: os.path.basename(x).split("-")[0]
            )
        ]
        all = []
        for i in by_case:
            count = os.path.basename(i[0]).split("-")[-1].split(".")[0]
            all.append(int(count))

        # Perform train/val/test splitting
        train, val, test = self.hparams.train_val_test_split  # type: ignore
        train_set, val_test_set, train_y, val_test_y = train_test_split(
            by_case,
            all,
            train_size=train,
            test_size=val + test,
            random_state=42,
        )

        val_set, test_set, v_y, t_y = train_test_split(
            val_test_set,
            val_test_y,
            test_size=test / (val + test),
            random_state=3,
        )

        # A utility function to convert a mask path to the corresponding
        # image path. This is used when we are not cropping to the bounding
        # box of the visual mention, and instead want to use the full image.
        def masksToImages(path):
            split_basename = os.path.basename(path).split("-")
            stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
            tensor_dir = os.path.dirname(os.path.dirname(path))
            img_path = tensor_dir + f"/images/{stem}"
            return img_path

        # Flatten cases into final lists. The size ratios will likely not be exact
        # to the desired ratios, but the goal is to get a relatively even amount
        # through random splitting.
        final_train_set = [slice for case in train_set for slice in case]
        final_test_set = [slice for case in test_set for slice in case]
        final_val_set = [slice for case in val_set for slice in case]

        # If we are not cropping to the bounding box of the visual mention,
        # convert all mask paths to image paths.
        if not self.hparams.crop:
            final_train_set = list(set([masksToImages(slice) for slice in final_train_set]))
            final_test_set = list(set([masksToImages(slice) for slice in final_test_set]))
            final_val_set = list(set([masksToImages(slice) for slice in final_val_set]))


        # Print out the number of classes in each split for sanity checking
        train_classes = [
            int(os.path.basename(i).split("-")[-1].split(".")[0])
            for i in final_train_set
        ]
        test_classes = [
            int(os.path.basename(i).split("-")[-1].split(".")[0])
            for i in final_test_set
        ]
        for i in set(all):
            print(i, train_classes.count(i))
            print(i, test_classes.count(i))

        # Create a weighted random sampler to assist with balancing the dataset
        # during training.
        c = []
        for i in final_train_set:
            count = os.path.basename(i).split("-")[-1].split(".")[0]
            c.append(int(count))

        weights = [0] * 20
        for i in set(c):
            weights[i] = 1 / c.count(i)
        sample_weights = [0] * len(c)
        for i in range(len(c)):
            sample_weights[i] = weights[c[i]]

        flat_val_test_set = [slice for case in val_test_set for slice in case]
        distinct_val_test_images = [
            list(i)
            for j, i in groupby(
                flat_val_test_set, lambda x: os.path.basename(x).split("-")[0:2]
            )
        ]
        new_dataset_size = (
            new_dataset_size - len(distinct_val_test_images)
        )

        self.sampler = WeightedRandomSampler(
            sample_weights, replacement=True, num_samples=new_dataset_size
        )

        # The WeightedRandomSampler does not support distributed training
        # out of the box, so we need to wrap it in a DistributedSamplerWrapper
        # to ensure each GPU gets a different subset of the data.
        if self.trainer.world_size > 1:
            self.distributed_sampler = DistributedSamplerWrapper(
                self.sampler, num_replicas=self.trainer.world_size, shuffle=False
            )
        else:
            self.distributed_sampler = self.sampler

        if self._train_dataset is None:
            # make training dataset
            self._train_dataset = VELCRODataset(
                items=final_train_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=False,
                from_mem=self.hparams.from_mem,
            )
        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = VELCRODataset(
                items=final_val_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=False,
                from_mem=self.hparams.from_mem,
            )
        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = VELCRODataset(
                items=final_test_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=True,
                from_mem=self.hparams.from_mem,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader. Only this
        dataloader is given a weighted sampler. Shuffling must be
        false when using the weighted random sampler.
        """
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self._train_device_batch_size,  # type: ignore
            sampler=self.distributed_sampler,
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.
        """
        assert self._val_dataset is not None
        sampler = None
        if self.trainer.world_size > 1:
            sampler = DistributedSampler(self._val_dataset)
        return DataLoader(
            self._val_dataset,
            batch_size=self._test_device_batch_size,  # type: ignore
            sampler=sampler,
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the test dataloader.
        """
        assert self._test_dataset is not None
        sampler = None
        if self.trainer.world_size > 1:
            sampler = DistributedSampler(self._test_dataset)
        return DataLoader(
            self._test_dataset,
            batch_size=self._test_device_batch_size,  # type: ignore
            sampler=sampler,
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
        )


class VELCRODataset(Dataset):
    """Dataset instance for a dataloader.

    Params:
        items (list[str]): A list of paths to the processed tensors.
        model_path (str): The huggingface path of the image tokenizer to use.
        is_testing (bool): Whether the dataset is being used for testing.
        from_mem (bool): Whether to load all data into memory. This can
            speed up training, but requires more memory.
    """

    def __init__(self, items: list[str], model_path: str, is_testing: bool, from_mem: bool):
        # assume our dataset contains image path, segmentation mask path, label
        self.items = items
        self.from_mem = from_mem
        if from_mem:
            self.examples = {idx: torch.load(path, weights_only=False) 
                             for idx, path in enumerate(items)}
        self.is_testing = is_testing
        # These transforms will always work with non-empty masks
        self.safe_transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
            ]
        )
        # These transforms may result in empty masks, so we need to
        # try them and revert if the mask is empty
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
        if self.from_mem:
            (img, label, candidate_text) = self.examples[idx]
        else:
            img = Image.open(self.items[idx])
            img = PILToTensor()(img)
            tensor_dir = os.path.dirname(os.path.dirname(self.items[idx]))
            (label, candidate_text, _) = torch.load(
                tensor_dir + f"/annotations/{os.path.splitext(
                    os.path.basename(self.items[idx]))[0]}.pt", weights_only=False
            )
        assert isinstance(img, torch.Tensor), f"{type(img)=}"
        #assert isinstance(mask, torch.Tensor), f"{type(mask)=}"
        assert isinstance(label, torch.Tensor), f"{type(label)=}"
        assert isinstance(candidate_text, dict), f"{type(candidate_text)=}"
        assert all(isinstance(x, torch.Tensor) for x in candidate_text.values())

        img = tv.Image(img)

        # Perform random transforms only during training
        if not self.is_testing:
            img = self.safe_transforms(img)
            try_img = self.danger_transforms(img)
            if torch.max(try_img) != 0:
                img = try_img

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

        return {
            "x": {
                "candidate_input": candidate_text,
                "image_input": {"img": img.pixel_values},
            },
            "y": {"class_indices": label, "path": self.items[idx]},
        }

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.items)
