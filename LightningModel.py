from argparse import ArgumentParser


from albumentations.augmentations.utils import read_rgb_image
import torch
import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader
from dataset import YOLODataset

import loss
import config
from model import *

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


class LitYolo(pl.LightningModule):
    def __init__(self, yolo, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.yolo = yolo
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        )

    def forward(self, x):
        return self.yolo(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = y[0], y[1], y[2]
        out = self.yolo(x)
        loss_fn = loss.YoloLoss()
        loss = (
            loss_fn(out[0], y0, self.scaled_anchors[0])
            + loss_fn(out[1], y1, self.scaled_anchors[1])
            + loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = y[0], y[1], y[2]
        out = self.yolo(x)
        loss_fn = loss.YoloLoss()
        loss = (
            loss_fn(out[0], y0, self.scaled_anchors[0])
            + loss_fn(out[1], y1, self.scaled_anchors[1])
            + loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        self.log('validation_loss', loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = y[0], y[1], y[2]
        out = self.yolo(x)
        loss_fn = loss.YoloLoss()
        loss = (
            loss_fn(out[0], y0, self.scaled_anchors[0])
            + loss_fn(out[1], y1, self.scaled_anchors[1])
            + loss_fn(out[2], y2, self.scaled_anchors[2])
        )
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitYolo")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser


class LitDataModule(pl.LightningDataModule):

    def __init__(self, train_csv_path, test_csv_path, batch_size: int = 32):
        super().__init__()
        IMAGE_SIZE = config.IMAGE_SIZE
        self.train_dataset = YOLODataset(
            train_csv_path,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        self.test_dataset = YOLODataset(
            test_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        self.train_eval_dataset = YOLODataset(
            train_csv_path,
            transform=config.test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=True,
            drop_last=False,
        )
        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
        return test_loader

    def test_dataloader(self):
        train_eval_loader = DataLoader(
            dataset=self.train_eval_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            shuffle=False,
            drop_last=False,
        )
        return train_eval_loader


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = LitYolo.add_model_specific_args(parser)
    parser = LitDataModule.add_argparse_args(parser)
    args = parser.parse_args()
    # ------------
    # data
    # ------------
    dm = LitDataModule(train_csv_path=config.DATASET +
                       "/train.csv", test_csv_path=config.DATASET + "/test.csv")

    # ------------
    # model
    # ------------
    model = LitYolo(YOLOv3(num_classes=config.NUM_CLASSES),
                    config.LEARNING_RATE)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(gpus='2')
    trainer.fit(model, datamodule=dm)
    # ------------
    # testing
    # ------------
    result = trainer.test(model, datamodule=dm)
    print(result)

# TODO Lightning Logs are running forewer because of huge yaml file generation
if __name__ == "__main__":
    cli_main()
