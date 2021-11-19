# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html

import multiprocessing
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class MNISTModel(LightningModule):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.l1 = torch.nn.Linear(n_input, n_hidden)
        self.relu = torch.nn.ReLU(inplace=True)
        self.l2 = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        input = x.view(x.size(0), -1)
        x1 = self.l1(input)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
        return DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
        return DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())


mnist_model = MNISTModel(28 * 28, 128, 10)

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
)

trainer.fit(mnist_model)
trainer.test(ckpt_path='best')

# export onnx file
# https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html
filepath = "./model.onnx"
input_sample = torch.randn((1, 28, 28))
mnist_model.to_onnx(filepath, input_sample, export_params=True)
