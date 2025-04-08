import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

import lib.medzoo as medzoo
# Lib files
import lib.utils as utils
from lib.losses3D import DiceLoss
from lib.losses3D.hausdorff_distance import compute_hd
from lib.medloaders import IXISegmentation
from lib.utils import prepare_input


def main():
    args = get_arguments()
    seed = 1777777
    utils.reproducibility(args, seed)

    included_files = get_included_files("../datasets/ixi_segmentation/original_data", "IOP")
    # included_files = ['189', '280', '504', '599', '608']

    test_dataset = IXISegmentation(mode='test', args=args)
    test_generator = DataLoader(test_dataset)

    model, optimizer = medzoo.create_model(args)

    criterion = DiceLoss(classes=args.classes)
    model.restore_checkpoint(args.pretrained)
    if args.cuda:
        model = model.to(device='mps')
        print("Model transferred in GPU.....")

    loss, std, per_channel_loss_mean, per_channel_loss_std, \
        hd_loss, hd_std, hd_loss_per_channel, hd_std_per_channel = infer_volume(model, criterion, test_generator, args)

    print(loss, std, per_channel_loss_mean, per_channel_loss_std,
          hd_loss, hd_std, hd_loss_per_channel, hd_std_per_channel)


def infer_volume(model, criterion, loader, args):
    model.eval()
    losses = []
    hd_loss_per_channel = []
    all_per_channel_loss = []

    for batch_idx, inputs in enumerate(loader):
        input_tuple = inputs[:-1]
        input_filename = inputs[-1]

        with torch.no_grad():
            input_tensor, target = prepare_input(input_tuple=input_tuple, args=args)
            input_tensor.requires_grad = False

            output = model(input_tensor)

            for i in range(output.shape[0]):
                save_images(output[i], input_filename[i].split(".")[0])

            loss, per_ch_score = criterion(output, target)

            for i in range(len(output)):
                hd_loss_per_channel.append(compute_hd(output[i], target[i]) / len(output))

            losses.append(loss.item())
            all_per_channel_loss.append(per_ch_score)

            print(f"The losses are: {loss}, per channel score: {per_ch_score}")

    losses = np.array(losses) / len(loader)
    final_loss = np.sum(losses)
    final_std = np.std(losses)

    hd_loss_per_channel = np.array(hd_loss_per_channel) / len(loader)

    final_hd_loss = np.sum(hd_loss_per_channel)
    final_hd_std = np.std(hd_loss_per_channel)

    final_hd_loss_per_channel = np.sum(hd_loss_per_channel, axis=0)
    final_hd_std_per_channel = np.std(hd_loss_per_channel, axis=0)

    print(len(loader))

    per_channel_loss_mean = np.mean(all_per_channel_loss, axis=0)
    per_channel_loss_std = np.std(all_per_channel_loss, axis=0)

    return (final_loss, final_std, per_channel_loss_mean, per_channel_loss_std,
            final_hd_loss, final_hd_std, final_hd_loss_per_channel, final_hd_std_per_channel)


def save_images(image, name):
    print(f"Now saving image: {name}")
    output_path = "../results/test_outputs"

    for index in range(image.shape[0]):
        slice = image[index].detach().cpu().numpy()
        slice = slice.squeeze()
        nib_image = nib.Nifti1Image(slice, affine=np.eye(4))
        nib.save(nib_image, os.path.join(output_path, name + "_" + str(index) + "_output.nii"))


def get_included_files(data_dir, hospital_name):
    files = [file for file in os.listdir(data_dir) if not file.startswith(".")]
    files = [file for file in files if hospital_name in file]

    extracted_ids, _ = get_values_from_pattern(files, pattern=r".*IXI(\d+)-.*\.nii", match_group=1)

    return extracted_ids


def get_values_from_pattern(list_of_string, pattern, match_group=1):
    extracted_nums = []
    extracted_strings = []

    for string in list_of_string:
        match = re.search(pattern, string)
        if match:
            id = match.group(match_group)
            if id not in extracted_nums:
                extracted_nums.append(id)
                extracted_strings.append(string)

    return extracted_nums, extracted_strings


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default="ixi_segmentation")
    parser.add_argument('--dim', nargs="+", type=int, default=(64, 64, 64))
    parser.add_argument('--nEpochs', type=int, default=250)
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--classes', type=int, default=3)
    parser.add_argument('--samples_train', type=int, default=1)
    parser.add_argument('--samples_val', type=int, default=1)
    parser.add_argument('--split', type=float, default=0.8)
    parser.add_argument('--inChannels', type=int, default=1)
    parser.add_argument('--inModalities', type=int, default=1)
    parser.add_argument('--normalization', default='full_volume_mean', type=str,
                        help='Tensor normalization: options ,max_min,',
                        choices=('max_min', 'full_volume_mean', 'brats', 'max', 'mean'))
    parser.add_argument('--augmentation', action='store_true', default=False)
    parser.add_argument('--fold_id', default='1', type=str, help='Select subject for fold validation')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='UNET3D',
                        choices=('VNET', 'VNET2', 'UNET3D', 'DENSENET1', 'DENSENET2', 'DENSENET3', 'HYPERDENSENET'))
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    parser.add_argument('--pretrained',
                        default='../saved_models/UNET3D_checkpoints/UNET3D_02_04___23_45_ixi_segmentation_/UNET3D_02_04___23_45_ixi_segmentation__last_epoch.pth',
                        type=str, metavar='PATH',
                        help='path to pretrained model')

    args = parser.parse_args()

    args.save = '../inference_checkpoints/' + args.model + '_checkpoints/' + args.model + '_{}_{}_'.format(
        utils.datestr(), args.dataset_name)
    args.tb_log_dir = '../runs/'
    return args


if __name__ == '__main__':
    main()

'''

def overlap_3d_image():
    B, C, D, H, W = 2, 1, 144, 192, 256
    #B, C, D, H, W = 1, 1, 4, 4, 4
    x = torch.randn(B, C, D, H, W)
    print('IMAGE shape ', x.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    kernel_size = 32
    stride = 16
    patches = x.unfold(4, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, H, num_of_patches_W, kernel_size]
    patches = patches.unfold(3, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, D, num_of_patches_H,num_of_patches_W, kernel_size,kernel_size]
    patches = patches.unfold(2, kernel_size, stride)
    print('patches shape ', patches.shape)  # [B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size]
    # patches = patches.unfold()
    # perform the operations on each patchff
    # ...
    B, C, num_of_patches_D, num_of_patches_H,num_of_patches_W, kernel_size ,kernel_size,kernel_size = patches.shape
    # # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    patches = patches.contiguous().view(B, C,num_of_patches_D* kernel_size, -1, kernel_size * kernel_size)
    print(patches.shape)
    print('slice shape ',patches[:,:,0,:,:].shape)
    slices = []
    for i in range(num_of_patches_D * kernel_size):

        output = F.fold(
              patches[:,:,i,:,:].contiguous().view(B, C * kernel_size * kernel_size,-1), output_size=(H, W), kernel_size=kernel_size, stride=stride)
        #print(output.shape)  # [B, C, H, W]
        slices.append(output)
    image = torch.stack(slices)
    print(image.shape)
    print(image.is_contiguous())
    image = image.permute(1,2,0,3,4).contiguous().view(B,C,-1,H*W)
    print(image.shape)
    output = F.fold(
        image.contiguous().view(B*H*W, C*kernel_size, -1), output_size=(D, 1), kernel_size=kernel_size, stride=stride)
    print(output.shape)  # [B, C, H, W]


'''
