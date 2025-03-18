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
from transformers import SamModel, SamProcessor

tqdm.pandas()

logger = RankedLogger(__name__)

# TODO(liamhebert): Throughout this code, we only tokenize the images in the
# "get_item" method. We should instead tokenize ahead of time.


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
        debug: bool = False,
        data_source: str = "ground_truths",
        crop: bool = False,
        from_mem = False,
        sam_masks: bool = False,
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
        os.makedirs(tensor_dir + "/images", exist_ok=True)

        #if self.hparams.sam_masks:
            #self.sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to("cuda")
            #self.sam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base") #.to("cuda")

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
                # throw_count+=1
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
            #elif np.max(masks[0]) == 255:
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
            #else:
                # Otherwise, match appropriate term to mask label for all masks.
                #candidate_mini_terms, masks = m_utils.match_term_mask(
                #    masks, imgs, potential_terms
                #)
                #candidate_terms = [
                #    umls_terms[term] for term in candidate_mini_terms
                #]

            for i, (img, mask, term) in enumerate(
                zip(imgs, masks, candidate_terms)
            ):

                y = term["idx"]
                candidate_text = term["desc"]
                mask = mask.astype(np.uint8)
                try:
                    label_ids = np.unique(mask)[1:]
                    if self.hparams.crop:
                        xs = []
                        ys = []
                        bboxes = regionprops(mask)
                        for prop in bboxes:
                            bbox = prop.bbox
                            prop_x = bbox[1]
                            prop_y = bbox[0]
                            prop_x2 = bbox[3]
                            prop_y2 = bbox[2]

                            xs.extend([prop_x, prop_x2])
                            ys.extend([prop_y, prop_y2])
                        max_x = max(xs)
                        max_y = max(ys)
                        min_x = min(xs)
                        min_y = min(ys)
                        cropped_img = img[
                            min_y:max_y,
                            min_x:max_x,
                        ]
                        if cropped_img.shape == img.shape:
                            print(min_x, max_x, min_y, max_y)
                        img = cropped_img
                    else:
                        #label_ids = np.unique(mask)[1:]
                        #if len(label_ids) > 20:
                        #    label_ids = label_ids[0:20]
                        #    print(len(label_ids))
                        img = Image.fromarray(img).convert("RGB")
                        img = img.resize((224, 224), Image.LANCZOS)
                        mask = Image.fromarray(mask)
                        img = PILToTensor()(img)

                        for k, label in enumerate(label_ids):
                            segment_mask = np.zeros_like(np.asarray(mask))
                            segment_mask[np.asarray(mask) == label] = label
                            y_indices, x_indices = np.where(segment_mask > 0)
                            x_scale = 224 / mask.size[0]
                            y_scale = 224 / mask.size[1]
                            input_box = [
                                int(np.min(x_indices)*x_scale), 
                                int(np.min(y_indices)*y_scale), 
                                int(np.max(x_indices)*x_scale), 
                                int(np.max(y_indices)*y_scale)
                            ]
                            #if self.hparams.sam_masks:
                                #inputs = self.sam_processor(img, input_boxes=[[input_box]], return_tensors="pt").to("cuda")
                                #with torch.no_grad():
                                #    outputs = sam_model(**inputs, multimask_output=False)
                                #segment_mask = self.sam_processor.image_processor.post_process_masks(
                                #           outputs.pred_masks.cpu(),
                                #           inputs["original_sizes"].cpu(),
                                #           inputs["reshaped_input_sizes"].cpu(),
                                #           binarize=True,
                                #)[0]
                                #print(type(segment_mask), segment_mask.shape)
                                #segment_mask = segment_mask.squeeze().numpy()
                            segment_mask = Image.fromarray(segment_mask)
                            #assert segment_mask.size == (224, 224), segment_mask.size
                            assert segment_mask.size == mask.size, segment_mask.size
                            segment_mask = segment_mask.resize((224, 224), Image.LANCZOS).convert("RGB")
                            segment_mask = PILToTensor()(segment_mask)
                            segment_mask = segment_mask.numpy()
                            segment_mask[segment_mask >= int(np.median(np.unique(segment_mask)))] = 1
                            segment_mask[segment_mask != 1] = 0
                            segment_mask = torch.from_numpy(segment_mask)

                            torch.save(
                                (segment_mask, y, candidate_text, input_box, True, segment_mask),
                                (f"{tensor_dir}/masks/{index}-{i}-{k}-{y}.pt"),
                            )
                        torch.save(
                            (img),
                            (f"{tensor_dir}/images/{index}-{i}-{y}.pt"),
                        )

                        try:
                            path = f"{tensor_dir}/masks/{index}-{i}-{k}-{y}.pt"
                            split_basename = os.path.basename(path).split("-")
                            stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
                            test_tensor_dir = os.path.dirname(os.path.dirname(path))
                            im_path = test_tensor_dir + f"/images/{stem}"
                            test = torch.load(im_path, weights_only=False)
                        except Exception as e:
                            print(e)
                            print(f"loading test failed! Tried to load {im_path}")

                except Exception as e:
                    print(f"Error on file when resizing: {row.File}")
                    print(img.shape, mask.shape)
                    print(e)

        logger.info("pre-tokenizing data....")

        Parallel(n_jobs=-1, backend="threading")(
            delayed(process)(row)
            for row in tqdm(mega.itertuples(index=False), total=len(mega))
        )
        #if self.hparams.sam_masks:
        #    self.generate_sam_masks()
        # print(throw_count)
        # mega.progress_apply(process, axis=1)

    def generate_sam_masks(self, mask_list):
        print("beginning sam mask generation!")
        sam_model = SamModel.from_pretrained("wanglab/medsam-vit-base") #.to("cuda:0")
        sam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
        full_iou = 0
        examples = list(glob(self.hparams.tensor_dir + "/masks/*.pt"))
        #print(len(mask_list))
        for i, mask_path in enumerate(mask_list):
            if i % 1000 == 0 and i != 0:
                print("current avg iou", full_iou/i)
            (mask, label, candidate_text, bb, valid, ground_truth) = torch.load(
                mask_path, weights_only=False
            )
            #path = mask_path
            #split_basename = os.path.basename(path).split("-")
            #stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
            #tensor_dir = os.path.dirname(os.path.dirname(path))
            #img = torch.load(tensor_dir + f"/images/{stem}", weights_only=False)
            #inputs = sam_processor(img, input_boxes=[[bb]], return_tensors="pt").to("cuda")
            #outputs = sam_model(**inputs, multimask_output=False)
            #segment_mask = sam_processor.image_processor.post_process_masks(
            #    outputs.pred_masks.cpu(),
            #    inputs["original_sizes"].cpu(),
            #    inputs["reshaped_input_sizes"].cpu(),
            #    binarize=True,
            #)[0]
            #segment_mask = segment_mask.squeeze().numpy()
            #valid = True
            #if len(list(np.unique(segment_mask))) < 2:
            #    valid = False
            #segment_mask = Image.fromarray(segment_mask)
            #assert segment_mask.size == (224, 224), segment_mask.size
            #segment_mask = segment_mask.convert("RGB")
            #segment_mask = PILToTensor()(segment_mask)
            #segment_mask[segment_mask != 0] = 1
            intersection = (mask * ground_truth).sum()
            #print(segment_mask.shape, ground_truth.shape)
            iou=None
            if intersection == 0:
                iou = 0.0
            else:
                union = torch.logical_or(mask, ground_truth).to(torch.int).sum()
                iou = intersection / union
            #print(iou)
            if iou > 1:
                print(i)
                #print(iou)
                #print(np.unique(mask.numpy()))
                path = mask_path
                split_basename = os.path.basename(path).split("-")
                stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
                tensor_dir = os.path.dirname(os.path.dirname(path))
                img = torch.load(tensor_dir + f"/images/{stem}", weights_only=False)
                inputs = sam_processor(img, input_boxes=[[bb]], return_tensors="pt") #.to("cuda")
                outputs = sam_model(**inputs, multimask_output=False)
                segment_mask = sam_processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu(),
                    binarize=True,
                )[0]
                segment_mask = segment_mask.squeeze().numpy()
                valid = True
                if len(list(np.unique(segment_mask))) < 2:
                    valid = False
                segment_mask = Image.fromarray(segment_mask)
                assert segment_mask.size == (224, 224), segment_mask.size
                segment_mask = segment_mask.convert("RGB")
                segment_mask = PILToTensor()(segment_mask)
                segment_mask[segment_mask != 0] = 1
                intersection = (segment_mask * ground_truth).sum()
                iou=None
                if intersection == 0:
                    iou = 0.0
                else:
                    union = torch.logical_or(mask, ground_truth).to(torch.int).sum()
                    iou = intersection / union
                assert iou <= 1
                torch.save(
                    (segment_mask, label, candidate_text, bb, valid, ground_truth),
                    (mask_path),
                )
            full_iou += iou
            #print(np.sum(segment_mask.numpy()), np.sum(ground_truth.numpy()))
            #print(np.unique(segment_mask.numpy()), np.unique(ground_truth.numpy()))
            #torch.save(
            #    (segment_mask, label, candidate_text, bb, valid, ground_truth),
            #    (mask_path),
            #)
        print("full iou:", full_iou/len(mask_list))


    def setup(self, stage: str):
        """Load dataset for training/validation/testing.

        NOTE: When using DDP (multiple GPUs), this is run once per GPU.
        As a result, this function should be deterministic and not download
        or have side effects. As a result, all data processing should be done in
        prepare_data and cached to disk, or done prior to training.

        Args:
            stage: either 'fit' (train), 'validate', 'test', or 'predict'
        """
        #if self.hparams.force_remake == False and self.hparams.sam_masks==True:
        #    self.generate_sam_masks()
        logger.info(f"Setting up data for stage: {stage}")

        # We only have access to trainer in setup, so we need to calculate
        # these parameters here.
        #print(self.trainer, self._train_device_batch_size, self.trainer.world_size)
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
            print("train device batch size", self._train_device_batch_size, self.trainer.world_size)
            self._test_device_batch_size = (
                self.hparams.test_batch_size // self.trainer.world_size
            )

        tensor_dir = self.hparams.tensor_dir  # type: ignore
        examples = list(glob(tensor_dir + "/masks/*.pt"))
        gliomas = []
        random.seed(42)
        all_check = []
        for i in examples:
            count = int(os.path.basename(i).split("-")[-1].split(".")[0])
            all_check.append(count)

        # Some classes are drastically overrepresented in the dataset.
        # In this case, some instances must be removed in order to ensure that
        # the testing and validation sets do not become too biased
        #removed_points = []
        #for i in set(all_check):
        #    if all_check.count(i) > 100000:
        #        print(i, all_check.count(i))
        #        reduced_sets = random.sample(
        #            [point for point in examples if int(os.path.basename(point).split("-")[-1].split(".")[0]) == i],
        #            int(all_check.count(i)/2)
        #        )
        #        removed_points.extend(reduced_sets)
        #print("number of datapoints to remove", len(removed_points))
        #examples = set(examples) - set(removed_points)
        #examples = list(examples)

        # This is similar to MedSAM's data processing approach, where they randomly select
        # one bounding box from the set of possible bounding boxes for a single datapoint.
        # The actual dataset size is only the amount of distinct images in the dataset
        new_dataset_size = int(len(list(glob(tensor_dir + "/images/*.pt"))))

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
        )
        # Flatten cases into final lists. The size ratios will likely not be exact
        # to the desired ratios, but the goal is to get a relatively even amount
        # through random splitting.
        final_train_set = [slice for case in train_set for slice in case]
        final_test_set = [slice for case in test_set for slice in case]
        final_val_set = [slice for case in val_set for slice in case]

        self.generate_sam_masks(final_test_set)
        #self.generate_sam_masks(final_val_set)

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
        print(new_dataset_size)
        distinct_val_test_images = [
            list(i)
            for j, i in groupby(
                flat_val_test_set, lambda x: os.path.basename(x).split("-")[0:2]
            )
        ]
        print(len(distinct_val_test_images))
        new_dataset_size = (
            new_dataset_size - len(distinct_val_test_images)
        )
        #new_dataset_size = len(examples)
        print(new_dataset_size)

        self.sampler = WeightedRandomSampler(
            sample_weights, replacement=True, num_samples=new_dataset_size
        )  # len(sample_weights))
        if self.trainer.world_size > 1:
            self.distributed_sampler = DistributedSamplerWrapper(
                self.sampler, num_replicas=self.trainer.world_size, shuffle=False
            )
        else:
            print("sampler gothere!")
            self.distributed_sampler = self.sampler

        if self._train_dataset is None:
            # make training dataset
            self._train_dataset = MedGeeseDataset(
                items=final_train_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=False,
            )
        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = MedGeeseDataset(
                items=final_val_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=False,
            )
        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = MedGeeseDataset(
                items=final_test_set,
                model_path=self.hparams.image_model_path,  # type: ignore
                is_testing=True,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
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
                #v2.PILToTensor(),
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
        (mask, label, candidate_text, _, valid, ground_truth) = torch.load(
                self.items[idx], weights_only=False
        )
        path = self.items[idx]
        split_basename = os.path.basename(path).split("-")
        stem = f"{split_basename[0]}-{split_basename[1]}-{split_basename[-1]}"
        tensor_dir = os.path.dirname(os.path.dirname(path))
        img = torch.load(tensor_dir + f"/images/{stem}", weights_only=False)

        assert isinstance(img, torch.Tensor), f"{type(img)=}"
        assert isinstance(mask, torch.Tensor), f"{type(mask)=}"
        assert isinstance(label, torch.Tensor), f"{type(label)=}"
        assert isinstance(candidate_text, dict), f"{type(candidate_text)=}"
        assert all(isinstance(x, torch.Tensor) for x in candidate_text.values())

        mask = tv.Mask(mask)
        img = tv.Image(img)
        if torch.max(mask) == 0 and valid:
            raise Exception("Empty mask pre")

        if not self.is_testing:
            img, mask = self.safe_transforms(img, mask)
            try_img, try_mask = self.danger_transforms(img, mask)
            if torch.max(try_mask) != 0 and valid:
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
        if torch.max(mask) == 0 and valid:
            raise Exception("Empty mask after")
        return {
            "x": {
                "candidate_input": candidate_text,
                "image_input": {"mask": mask, "img": img.pixel_values},
            },
            "y": {"class_indices": label, "valid_mask": valid, "mask_gt": ground_truth},
        }

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.items)
