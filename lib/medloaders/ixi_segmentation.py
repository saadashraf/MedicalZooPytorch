import glob
import os

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import lib.augment3D as augment3D
import lib.utils as utils
from lib.medloaders import medical_image_process as img_loader
from lib.medloaders.medical_image_process import load_medical_image
from lib.medloaders.medical_loader_utils import get_viz_set, create_sub_volumes


class IXISegmentation(Dataset):
    """
    Code for reading the IXI brain MRI segmentation dataset, sourced from - https://www.kaggle.com/datasets/hamedamin/preprocessed-oasis-and-epilepsy-and-ixi/data
    """

    def __init__(self, mode, args, included_files=None, dataset_path='../datasets/ixi_segmentation'):
        """
        :param mode: 'train','val','test'
        :param dataset_path: root dataset folder
        """
        self.mode = mode
        self.root = str(dataset_path)
        self.classes = args.classes
        # self.threshold = args.threshold
        self.normalization = args.normalization
        self.augmentation = args.augmentation
        self.list = []
        if self.augmentation:
            self.transform = augment3D.RandomChoice(
                transforms=[augment3D.GaussianNoise(mean=0, std=0.01), augment3D.RandomFlip(),
                            augment3D.ElasticTransform()], p=0.5)

        if self.mode == "train":
            self.dataset_path = os.path.join(self.root, "train")

        elif self.mode == 'val':
            self.dataset_path = os.path.join(self.root, "val")

        else:
            self.dataset_path = os.path.join(self.root, "test")

        self.full_volume = None
        self.affine = None

        self.input_dataset = os.listdir(os.path.join(self.dataset_path, "scan"))
        self.input_dataset = [data for data in self.input_dataset if data.endswith(".nii")
                              and (included_files is None or any(sub in data for sub in included_files))]
        self.wm_label_dataset = os.listdir(os.path.join(self.dataset_path, "wm_mask"))
        self.wm_label_dataset = [data for data in self.wm_label_dataset if data.endswith(".nii")
                                 and (included_files is None or any(sub in data for sub in included_files))]
        self.gm_label_dataset = os.listdir(os.path.join(self.dataset_path, "gm_mask"))
        self.gm_label_dataset = [data for data in self.gm_label_dataset if data.endswith(".nii")
                                 and (included_files is None or any(sub in data for sub in included_files))]

        print(len(self.input_dataset))

    def __len__(self):
        return len(self.input_dataset)

    def __getitem__(self, index):
        input_scan = nib.load(os.path.join(self.dataset_path, "scan", self.input_dataset[index])).get_fdata()
        wm_label_scan = nib.load(os.path.join(self.dataset_path, "wm_mask", self.wm_label_dataset[index])).get_fdata()
        gm_label_scan = nib.load(os.path.join(self.dataset_path, "gm_mask", self.gm_label_dataset[index])).get_fdata()

        filename = self.input_dataset[index]

        wm_label_scan = torch.from_numpy(wm_label_scan)
        gm_label_scan = torch.from_numpy(gm_label_scan)

        label_mask = self.create_multiclass_mask(wm_label_scan, gm_label_scan)

        label_mask = F.one_hot(label_mask, num_classes=self.classes).permute(3, 0, 1, 2)

        if self.augmentation:
            augmented_scan = self.transform(input_scan)

            return torch.tensor(augmented_scan.copy(), dtype=torch.float32).unsqueeze(0), label_mask.float()

        output_path = "../results/output_images"

        if index % 100 == 0:
            for slice_index in range(label_mask.shape[0]):
                image = label_mask[slice_index].detach().cpu().numpy()
                image = image.squeeze()
                nib_image = nib.Nifti1Image(image, affine=np.eye(4), dtype="int64")
                nib.save(nib_image, os.path.join(output_path, f"test_mask_{index}_{slice_index}" + ".nii"))

        return torch.tensor(input_scan, dtype=torch.float32).unsqueeze(0), label_mask.float(), filename

    def create_multiclass_mask(self, wm_mask, gm_mask):
        wm_mask = (wm_mask > 0.1).long()
        gm_mask = (gm_mask > 0.1).long()

        multi_mask = torch.zeros_like(wm_mask, dtype=torch.long)

        multi_mask[wm_mask == 1] = 1
        multi_mask[gm_mask == 1] = 2

        return multi_mask
