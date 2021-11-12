# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html

import multiprocessing
import os

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    # https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html
    def test_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=multiprocessing.cpu_count())

# Initialize a trainer
trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=3,
)

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

trainer.test(dataloaders=test_loader)

# https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html
filepath = "model.onnx"
input_sample = torch.randn((1, 28, 28))
mnist_model.to_onnx(filepath, input_sample, export_params=True)
