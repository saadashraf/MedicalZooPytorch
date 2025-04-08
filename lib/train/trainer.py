from typing import Optional

import os
import numpy as np
import torch
import nibabel as nib

from lib.utils.early_stopping import EarlyStopping
from lib.utils.general import prepare_input
from lib.visual3D_temp.BaseWriter import TensorboardWriter


class Trainer:
    """
    Trainer class
    """

    def __init__(self, args, model, criterion, optimizer, train_data_loader, valid_data_loader=None, lr_scheduler=None):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data_loader = train_data_loader

        self.early_stopping_patience = args.early_stopping_patience

        self.early_stopping = (
            EarlyStopping(patience=self.early_stopping_patience)
            if self.early_stopping_patience
            else None
        )


        # epoch-based training
        self.len_epoch = len(self.train_data_loader)
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(train_data_loader.batch_size))
        self.writer = TensorboardWriter(args)

        self.save_frequency = 10
        self.terminal_show_freq = self.args.terminal_show_freq
        self.start_epoch = 1

    def training(self):
        for epoch in range(self.start_epoch, self.args.nEpochs):
            self.train_epoch(epoch)

            if self.do_validation:
                self.validate_epoch(epoch)

            val_loss = self.writer.data['val']['loss'] / self.writer.data['val']['count']

            if self.args.save is not None and ((epoch + 1) % self.save_frequency):
                self.model.save_checkpoint(self.args.save,
                                           epoch, val_loss,
                                           optimizer=self.optimizer)

            self.writer.write_end_of_epoch(epoch)

            self.writer.reset('train')
            self.writer.reset('val')

            if self.early_stopping:
                self.early_stopping(val_loss=val_loss)

                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

    def train_epoch(self, epoch):
        self.model.train()

        for batch_idx, inputs in enumerate(self.train_data_loader):

            self.optimizer.zero_grad()

            input_tuple = inputs[:-1]
            input_filename = inputs[-1]

            input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            loss_dice, per_ch_score = self.criterion(output, target)
            loss_dice.backward()
            self.optimizer.step()

            self.writer.update_scores(batch_idx, loss_dice.item(), per_ch_score, 'train',
                                      epoch * self.len_epoch + batch_idx)

            if (batch_idx + 1) % self.terminal_show_freq == 0:
                partial_epoch = epoch + batch_idx / self.len_epoch - 1
                self.writer.display_terminal(partial_epoch, epoch, 'train')

        self.writer.display_terminal(self.len_epoch, epoch, mode='train', summary=True)

    def validate_epoch(self, epoch):
        self.model.eval()

        for batch_idx, input_tuple in enumerate(self.valid_data_loader):
            with torch.no_grad():
                input_tensor, target = prepare_input(input_tuple=input_tuple, args=self.args)
                input_tensor.requires_grad = False

                output = self.model(input_tensor)

                if epoch % 5 == 0:
                    self.save_images(output[0], str(epoch))

                loss, per_ch_score = self.criterion(output, target)

                self.writer.update_scores(batch_idx, loss.item(), per_ch_score, 'val',
                                          epoch * self.len_epoch + batch_idx)

        self.writer.display_terminal(len(self.valid_data_loader), epoch, mode='val', summary=True)

    def save_images(self, image, name):
        print(f"Now saving image: {image}")
        output_path = "results/output_images"

        for index in range(image.shape[0]):
            slice = image[index].detach().cpu().numpy()
            slice = slice.squeeze()
            nib_image = nib.Nifti1Image(slice, affine=np.eye(4))
            nib.save(nib_image, os.path.join(output_path, name + f"_{index}.nii"))
