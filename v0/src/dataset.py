# so tokenizer works. Now we will create a dataset class to hold everything
# for some reason this is the most confusing part for me rip
import numpy as np
from PIL import Image, ImageFile
import PIL
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPImageProcessor
from glob import glob
from tqdm import tqdm
import os
import torch
import json
import random
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.tv_tensors as tv

tqdm.pandas()
# Our medical dataset will contain images and their masks collected from multiple sources


class medicalDataModule(LightningDataModule):
    def prepare_data(self) -> None:

        datasets = [
            "labels_calc_mammograms.csv",
            "labels_mass_mammograms.csv",
            "labels_calc_mammograms_test.csv",
            "labels_mass_mammograms_test.csv",
            "duke_breast_cancer_annotations.csv",
            "liver_dataset_fixed_trimmed.csv",
        ]
        # get umls master dict: TODO: reorganize so it is easier to get specific terms
        with open(self.data_dir + "/" + "UMLS_formatted.json") as json_file:
            umls_terms = json.load(json_file)

        # this is so bad but i rlly wanna get the multi-label mask thing going ;_;
        dataset_dataframes = [
            (dataset, pd.read_csv(self.data_dir + "/" + dataset))
            for dataset in datasets
        ]
        if self.debug:
            dataset_dataframes = [
                (dataset, x.iloc[:1000]) for dataset, x in dataset_dataframes
            ]
        # manually process liver

        _, mega = dataset_dataframes[-1]  # the liver dataset
        mega.dropna(subset="cancer", inplace=True)
        mega.rename(
            columns={
                "liver": "organ",
                "cancer": "abnormality_type",
                "slice_id": "patient_id",
            },
            inplace=True,
        )
        mega.replace({"organ": {0: "Background", 1: "liver"}}, inplace=True)
        mega.replace(
            {"abnormality_type": {0: "none", 1: "tumor"}}, inplace=True
        )

        mega.to_csv("mega_liver.csv")

        # mass and calc datasets
        for dataset, ds in dataset_dataframes[:-2]:
            # renamed but columns not modified
            ds = ds.rename(
                columns={
                    "image file path": "image",
                    "cropped image file path": "cropped",
                    "ROI mask file path": "mask",
                    "abnormality type": "abnormality_type",
                }
            )
            # if malignant mass detected, change to tumor
            ds = ds[
                ["image", "mask", "abnormality_type", "pathology"]
            ]  # keep just what we need
            # we dont care if it's malignant or benign, just if it's a tumor or not
            if "mass" in dataset:
                ds["abnormality_type"] = "tumor_mass"
            elif "calc" in dataset:
                ds["abnormality_type"] = "tumor_calc"

            # ds.loc[ds['pathology'] == 'BENIGN', 'abnormality_type'] = 'benign'

            mega = pd.concat([mega, ds])

        # duke dataset
        _, ds = dataset_dataframes[-2]
        mega = pd.concat([mega, ds])  # should just work out of the box

        # heres where the modifications begin
        mega["organ"] = mega["organ"].fillna("breast")

        # just take the columns we need. if we want to keep all of them then skip this part
        concise = mega[
            ["train", "image", "mask", "organ", "abnormality_type"]
        ].copy()
        # looks like the batches are dominated by background images so removing for now
        # concise = concise[(concise.organ != 'none') & (concise.abnormality_type != 'none')]
        concise["index"] = 1
        concise["index"] = concise["index"].cumsum() - 1  # 0, 1, 2, 3 etc.

        os.makedirs(
            self.tensor_dir + "/processed_tensors", exist_ok=True
        )  # NOTE: This is giga slow because /mnt/DATA is a HDD drive, not SSD, we should move everything off this drive
        # once the tensors are on the SSD drive, it runs at 206it/s per batch of 32

        # def make_one_hot(umls_terms):
        #     labels = list(umls_terms.values())
        #     int_labels = torch.arange(0, len(labels))
        #     self.encoding_dict = {k:v for k,v in zip(labels,int_labels)}

        # def encode_one_hot(terms):
        #     # take in a list of terms and return onehot encodings
        #     one_hots = []
        #     for term in terms:
        #         one_hot = torch.zeros(len(self.encoding_dict.keys())-1)
        #         idx = self.encoding_dict[term]
        #         if idx != -1:
        #             one_hot[idx] = 1
        #         one_hots.append(one_hot)
        #     return one_hots

        # make_one_hot(umls_terms)
        def process(row):
            if (
                os.path.exists(
                    self.tensor_dir
                    + "/processed_tensors/"
                    + str(row["index"])
                    + "-0.pt"
                )
                and self.force_remake == False
            ):
                return
            image_path = self.image_dir + "/" + str(row["image"])
            mask_path = self.image_dir + "/" + str(row["mask"])
            if (
                os.path.exists(image_path) == False
                or os.path.exists(mask_path) == False
            ):
                print("MISSING FILE: ", image_path, mask_path)
                return

            label = row["abnormality_type"]
            is_breast = "png" in image_path
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
            if not is_breast:
                img = img.convert("RGB").resize(
                    (224, 224), Image.LANCZOS
                )  # only liver ones are not scaled
                mask = np.array(
                    mask.convert("RGB").resize((224, 224), Image.LANCZOS)
                )
                mask_arrays = [
                    np.copy(mask) for _ in range(len(np.unique(mask)))
                ]
                # make first mask only background -> we can def do this better but just wanna make sure we get functionality 1st
                # Note: for background the white pixels will be the background pixels, black are non-background
                # bground 1st, then liver, then tumor
                # print('ITER', masks, np.unique(mask), ['background', 'Liver', 'Malignant neoplasm of liver'])
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
                    # print("ADDED MASK")
                    masks.append(mask_kind)
            else:
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
                # elif label == "benign":
                #     umls_labels.append(umls_terms['Benign neoplasm of breast'])
                else:
                    raise Exception("Label not recognized", label)

            # one_hots = encode_one_hot(umls_labels) # ??

            labels = torch.tensor(umls_labels)

            # mask_features = self.processor(images=masks, return_tensors="pt", do_normalize=False, do_rescale=False, do_center_crop=False,  do_resize=False)

            # masks = torch.stack([v2.functional.pil_to_tensor(x).float() for x in masks])
            # try:
            #     masks = self.expand_embedding_blowup_mask(masks) > 0
            # except RuntimeError as e:
            #     print('ERRORED', masks.shape)
            #     raise e

            # mask_features['pixel_values'] = mask_features['pixel_values'].float()

            # if len(labels) > 1 or len(labels) != len(mask):
            #     print(labels)
            for i, (m, y) in enumerate(zip(masks, labels)):
                index = str(row["index"])

                if np.max(np.array(m)) == 0:
                    print("skipping empty mask")
                    continue
                # m = v2.functional.to_pil_image(m.int())
                torch.save(
                    (img, m, y, row["organ"]),
                    f"{self.tensor_dir}/processed_tensors/{index}-{str(i)}.pt",
                )

        print("pre-tokenizing data....")
        concise.progress_apply(
            process, axis=1
        )  # this takes about 2 hrs to run on the whole dataset, but we only need to do it once

    def setup(self, stage=None):
        print(stage)

        examples = glob(self.tensor_dir + "/processed_tensors/*.pt")
        total = len(examples)
        # train test split
        # TODO: use sklearn train-test-split
        # random.shuffle(examples)
        train = examples[: int(0.8 * total)]
        val = examples[int(0.8 * total) : int(0.9 * total)]
        test = examples[int(0.9 * total) :]

        self.train_dataset = medicalDataset(train, self.model_path)
        self.val_dataset = medicalDataset(val, self.model_path)
        self.test_dataset = medicalDataset(test, self.model_path)

    def __init__(
        self,
        model_path="openai/clip-vit-base-patch16",
        data_dir: str = "data",
        batch_size: int = 32,
        image_dir: str = "data/images-scaled-224/",
        tensor_dir: str = "local_tensors_dir/",
        debug=False,
        force_remake=False,
        mask_kernel="square",
        local_machine=False,
    ):
        ###
        # data_dir: directory containing the csv files
        # batch_size: batch size for training and validation
        # image_dir: directory containing the images
        # tensor_dir: directory containing the pre-processed tensors, should be on local SSD not /mnt/DATA
        ###

        super().__init__()
        self.prepare_data_per_node = False
        self.data_dir = data_dir
        self.debug = debug
        self.batch_size = batch_size
        self.image_dir = image_dir
        self.tensor_dir = tensor_dir
        self.model_path = model_path
        self.force_remake = force_remake
        self.mask_kernel = mask_kernel
        self.local_machine = local_machine

        # self.processor = CLIPImageProcessor.from_pretrained(self.model_path, local_files_only=False) # set this to False if you want to download the tokenizer, you only need to do this once
        # visionmodel = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", local_files_only=False)
        # print('VISION CONV',visionmodel.vision_model.embeddings.patch_embedding)

        if local_machine:
            self.tensor_dir = "/tmp/" + tensor_dir
        if "patch" not in model_path:
            raise NotImplementedError("model_path must contain patch size")

        self.save_hyperparameters()

        # examples = glob(self.tensor_dir + '/processed_tensors/*.pt')
        # total = len(examples)
        # #train test split
        # # TODO: use sklearn train-test-split
        # random.shuffle(examples)
        # train = examples[:int(0.8*total)]
        # val = examples[int(0.8*total):int(0.9*total)]
        # test = examples[int(0.9*total):]

        # self.train_dataset = medicalDataset(train)
        # self.val_dataset = medicalDataset(val)
        # self.test_dataset = medicalDataset(test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=6,
            persistent_workers=False,
        )


class medicalDataset(Dataset):
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
        self.processor = CLIPImageProcessor.from_pretrained(
            model_path, local_files_only=False
        )  # set this to False if you want to download the tokenizer, you only need to do this once

    def __getitem__(self, idx):
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
        return len(self.items)


if __name__ == "__main__":
    db = medicalDataModule(debug=False)
    # db.setup()

    train = db.train_dataloader()
    for x, mask in tqdm(train):
        ...
