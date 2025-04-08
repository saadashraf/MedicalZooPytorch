import numpy as np
import torch
from medpy.metric.binary import hd95
from monai.metrics import HausdorffDistanceMetric
from scipy.spatial.distance import directed_hausdorff


def compute_hd(pred_mask, gt_mask, threshold=0.1):
    """
    Compute Hausdorff 95 Distance (HD95) for multi-class segmentation.

    Args:
        pred_mask (torch.Tensor): (C, H, W, D) soft predictions or one-hot encoded.
        gt_mask (torch.Tensor): (C, H, W, D) one-hot encoded ground truth.
        threshold (float): Threshold for binarizing predictions.
        spacing (tuple): Voxel spacing (e.g., from NIfTI header).

    Returns:
        dict: {class_index: HD95 value}
    """

    num_classes = pred_mask.shape[0]

    pred_mask = pred_mask.cpu().detach().numpy()
    gt_mask = gt_mask.cpu().detach().numpy()

    final_hd_values = np.zeros((num_classes - 1, 1))

    for cls in range(1, num_classes):
        pred_bin = (pred_mask[cls] >= threshold).astype(np.bool_)
        gt_bin = (gt_mask[cls] >= threshold).astype(np.bool_)

        hd_values = []

        for index in range(pred_bin.shape[0]):
            pred_bin_slice = pred_bin[index]
            gt_bin_slice = gt_bin[index]

            hd_values.append(directed_hausdorff(pred_bin_slice, gt_bin_slice)[0])

        final_hd_values[cls - 1] = np.mean(np.array(hd_values))

    return final_hd_values
