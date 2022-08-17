import os

import pytorch_lightning as pl
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from mnist_gan.types import *


class MNISTDataModule(pl.LightningDataModule):
    MNIST_STD = 0.3081
    MNIST_MEAN = 0.1037

    def __init__(
        self,
        data_dir,
        train_val_ratio=[0.8, 0.2],
        batch_size=32,
        num_workers=0,
        *args, **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            ignore=['func'],
        )
        self.data_dir = data_dir
        self.train_val_ratio = train_val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.MNIST_MEAN,), (self.MNIST_STD,)),
        ])
        self.target_transform = torch.nn.functional.one_hot

    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_split = MNIST(
                root=self.data_dir,
                train=True,
                download=False,
                transform=self.transforms,
                target_transform=lambda x: self.target_transform(torch.tensor(x), 10).float(),
            )
            train_set_size = len(self.train_split) * self.train_val_ratio[0] / sum(self.train_val_ratio)
            train_set_size = int(train_set_size)
            valid_set_size = len(self.train_split) - train_set_size
            self.train_split, self.valid_split = random_split(
                dataset=self.train_split,
                lengths=[train_set_size, valid_set_size],
                generator=torch.Generator().manual_seed(int(os.environ.get("PL_GLOBAL_SEED"))),
            )
        if stage == "test" or stage is None:
            self.test_split = MNIST(
                root=self.data_dir,
                train=False,
                download=False,
                transform=self.transforms,
                target_transform=lambda x: self.target_transform(torch.tensor(x), 10).float(),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_argparse_args(parser: ArgumentParser) -> ArgumentParser:
        data_parser = parser.add_argument_group('Dataset')
        data_parser.add_argument('--data_dir', type=Path, default='./data_root')
        data_parser.add_argument('--batch_size', type=int, default=32)
        data_parser.add_argument('--num_workers', type=int, default=0)
        return parser

    @classmethod
    def to_image(cls, tensors):
        tensors = tensors * cls.MNIST_STD + cls.MNIST_MEAN 
        tensors = tensors.view(-1, 28, 28)
        images = [transforms.ToPILImage()(tensor) for tensor in tensors]

        return images