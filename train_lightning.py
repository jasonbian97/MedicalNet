import pytorch_lightning as pl
from torch.optim import Adam
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateLogger
import os
os.chdir(os.path.dirname(__file__)) # set current .py file as working directory
import sys
from scipy import ndimage
from torch import optim
import numpy as np
import torch.nn.functional as F
from test import seg_eval
from datasets.BiMaskDataset import TriPairDataset
from setting import parse_opts
from model import generate_model
# from test import seg_eval
import nibabel as nib
from easydict import EasyDict as edict

class BiMaskSeg(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()

        # do this to save all arguments in any logger (tensorboard)
        # hparams = edict(hparams) # add this line when testing
        self.hparams = hparams

        with open(hparams.dataset_info,"r") as fp:
            self.dataset_info = json.load(fp)
            self.dataset_info = self.dataset_info[hparams.dataset]

        # getting model
        torch.manual_seed(hparams.manual_seed)
        self.model, self.parameters_dict = generate_model(hparams)
        # pay attention to the unbalanced samples. ignore the bg here (idx = 0)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1,88.,8.6,17.7,3.3,19.3,51.7]).cuda())
        print(self.model)

    def train_dataloader(self):
        training_dataset = TriPairDataset(self.dataset_info, self.hparams, mode="train",
                                         option = self.hparams.dataset_option)
        data_loader = DataLoader(training_dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                 num_workers=self.hparams.num_workers, pin_memory=True)
        return data_loader


    def val_dataloader(self):
        testing_data = TriPairDataset(self.dataset_info, self.hparams, mode="val",
                                     option = self.hparams.dataset_option)
        data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=False)
        return data_loader

    def test_dataloader(self):
        testing_data = TriPairDataset(self.dataset_info, self.hparams, mode="val",
                                     option = self.hparams.dataset_option)
        data_loader = DataLoader(testing_data, batch_size=1, shuffle=False, num_workers=6, pin_memory=False)
        return data_loader

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        params = [
            {'params': self.parameters_dict['base_parameters'], 'lr': self.hparams.learning_rate},
            {'params': self.parameters_dict['new_parameters'], 'lr': self.hparams.learning_rate * 100}
        ]
        optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones)
        return {"optimizer":optimizer,"lr_scheduler":scheduler}

    def training_step(self, batch, batch_idx):
        volumes, label_masks = batch
        out_masks = self(volumes) #(bs,num_calss,D,H,W)

        # resize label
        [n, _, d, h, w] = out_masks.shape
        new_label_masks = np.zeros([n, d, h, w])
        for label_id in range(n):
            label_mask = label_masks[label_id]
            [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
            label_mask = np.reshape(label_mask.cpu(), [ori_d, ori_h, ori_w])
            scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
            label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
            new_label_masks[label_id] = label_mask

        new_label_masks = torch.tensor(new_label_masks).to(torch.int64)

        new_label_masks = new_label_masks.cuda()
        loss = self.criterion(out_masks, new_label_masks)

        # add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        volume, label_masks, ori_mask_arr, _  = batch

        out_masks = self(volume)

        # resize label
        [n, _, d, h, w] = out_masks.shape
        new_label_masks = np.zeros([n, d, h, w])
        for label_id in range(n):
            label_mask = label_masks[label_id]
            [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
            label_mask = np.reshape(label_mask.cpu(), [ori_d, ori_h, ori_w])
            scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
            label_mask = ndimage.interpolation.zoom(label_mask, scale, order=0)
            new_label_masks[label_id] = label_mask

        new_label_masks = torch.tensor(new_label_masks).to(torch.int64)
        new_label_masks = new_label_masks.cuda()
        loss = self.criterion(out_masks, new_label_masks)

        probs = F.softmax(out_masks, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        [batchsize, depth, height, width] = ori_mask_arr.shape
        mask = probs[0]
        scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
        mask = ndimage.interpolation.zoom(mask.cpu(), scale, order=1)
        mask = np.argmax(mask, axis=0)
        ori_mask = ori_mask_arr[0]
        return {'val_loss': loss, "pred":mask, "label":ori_mask}

    def test_step(self,  batch, batch_idx):
        volume, label_masks, ori_mask_arr,affine = batch
        out_masks = self(volume)

        probs = F.softmax(out_masks, dim=1)

        # resize mask to original size
        [batchsize, _, mask_d, mask_h, mask_w] = probs.shape
        [batchsize, depth, height, width] = ori_mask_arr.shape
        mask = probs[0]
        scale = [1, depth * 1.0 / mask_d, height * 1.0 / mask_h, width * 1.0 / mask_w]
        mask = ndimage.interpolation.zoom(mask.cpu(), scale, order=1)
        mask = np.argmax(mask, axis=0)
        ori_mask_arr = ori_mask_arr[0]

        # write image to niigz
        mask = mask.astype(np.uint8)
        new_img = nib.Nifti1Image(mask, affine[0].cpu())
        nib.save(new_img, "trails/logs/train_lightning/val_img_{}.nii.gz".format(batch_idx))
        return {"pred": mask, "label": ori_mask_arr}


    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        pred_total = [x['pred'] for x in outputs]
        y_total = [x['label'].cpu() for x in outputs]
        Nimg = len(pred_total)
        dices = np.zeros([Nimg, self.hparams.n_seg_classes])
        for idx in range(Nimg):
            dices[idx, :] = seg_eval(pred_total[idx], y_total[idx], range(self.hparams.n_seg_classes))
        print("avg_loss(val) = ",avg_loss)
        print("dice_each_class(val) = ",dices.mean(axis=0))
        self.logger.experiment.add_text("dice_each_class", str(dices.mean(axis=0)), self.current_epoch)
        # print(self.trainer.lr_schedulers.get_lr())
        mDice = dices.mean()
        logs = {"val_loss":avg_loss,"mDice":torch.tensor(mDice)}
        return {'log': logs}



if __name__ == '__main__':
    # settings
    hparams = parse_opts()

    checkpoint_callback = ModelCheckpoint(
        filepath=None,
        monitor='mDice',
        save_top_k=1,
        verbose=True,
        mode='max'
    )

    lr_logger = LearningRateLogger()

    if hparams.phase == "test":
        pretrained_model = BiMaskSeg.load_from_checkpoint(
            checkpoint_path = "trails/logs/train_lightning/lightning_logs/version_0/checkpoints/epoch=29.ckpt",
            hparams_file= "trails/logs/train_lightning/lightning_logs/version_0/hparams.yaml"
        )
        trainer = Trainer(gpus=hparams.gpu_id)
        trainer.test(pretrained_model)
        exit(0)

    Sys = BiMaskSeg(hparams=hparams)
    trainer = Trainer(checkpoint_callback=checkpoint_callback,
                      callbacks=[lr_logger],
                      gpus=hparams.gpu_id,
                      default_root_dir='results/{}'.format(os.path.basename(__file__)[:-3]),
                      max_epochs = hparams.n_epochs,
                      check_val_every_n_epoch=25,
                      val_percent_check=0.5
                      )

    trainer.fit(Sys)
