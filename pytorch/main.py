# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/mnist-hello-world.html

import multiprocessing
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
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

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        input = x.view(x.size(0), -1)
        x1 = self.l1(input)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        preds = self(x)
        acc = self.accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc_step', acc)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())

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
    max_epochs=30,
)

trainer.fit(mnist_model)
trainer.test(ckpt_path='best')

# export onnx file
# https://pytorch-lightning.readthedocs.io/en/latest/common/production_inference.html
filepath = "./model.onnx"
input_sample = torch.randn((1, 28, 28))
mnist_model.to_onnx(filepath, input_sample, export_params=True)

print("predict sample start...")
test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
data = test_ds[0]
predtest_input = torch.zeros(1, 28, 28)
label = data[1]
print(f"predtest_input = {predtest_input}, label = {label}")
pred = mnist_model(predtest_input)
print(f"pred = {pred}")
result = torch.nn.Softmax(dim=1)(pred)
print(f"result = {result}")

import numpy as np

# <print(ndarr)> | sed -E 's/0\./0,/g' | sed -E 's/([1-9]) /\1, /g' | sed -E 's/0,([0-9])/0.\1/g' | sed -E 's/([1-9])$/\1,/g' | sed -E 's/]$/],/g' | clip.sh
arr =  [[[0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.06666667, 0.21960784, 0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.54509807, 0.7058824,  0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.7607843,  0.40784314, 0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0.39607844, 0.7647059,  0.12156863, 0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.79607844, 0.2901961,  0,         0,         0,
   0.01960784, 0.3882353,  0.06666667, 0,         0,         0,
   0,         0,         0,         0,         0.00784314, 0.5764706,
   0.9019608,  0.3372549,  0.01960784, 0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.78431374, 0.24705882, 0,         0,         0,
   0.2901961,  0.9411765,  0.11764706, 0,         0,         0,
   0,         0,         0,         0.01568628, 0.654902,   0.8392157,
   0.1254902,  0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.78431374, 0.24705882, 0,         0,         0,
   0.7490196,  0.95686275, 0.09019608, 0,         0,         0,
   0,         0,         0,         0.61960787, 0.80784315, 0.08235294,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.78431374, 0.24705882, 0,         0,         0.28627452,
   0.91764706, 0.92941177, 0.23137255, 0,         0,         0,
   0,         0,         0.4862745,  0.8784314,  0.11372549, 0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.7921569,  0.27058825, 0,         0.05098039, 0.83137256,
   0.43137255, 0.654902,   0.53333336, 0,         0,         0,
   0,         0.36078432, 0.92941177, 0.21176471, 0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.7607843,  0.49803922, 0,         0.5803922,  0.7882353,
   0,         0.3764706,  0.8156863,  0.01960784, 0,         0,
   0.38431373, 0.94509804, 0.3137255,  0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0.27450982, 0.8901961,  0.89411765, 0.85882354, 0.17254902,
   0,         0.06666667, 0.85490197, 0.49411765, 0.10980392, 0.5568628,
   0.9254902,  0.2901961,  0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0.07843138, 0.23137255, 0.09019608, 0,
   0,         0,         0.23137255, 0.8392157,  0.8901961,  0.75686276,
   0.2,        0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0.04313726, 0.09019608, 0.01176471,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ],
  [0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,         0,         0,
   0,         0,         0,         0,        ]]],


sample = np.array(arr)
input = torch.from_numpy(sample.astype(np.float32)).clone()
pred = mnist_model(input)
print(f"pred = {pred}")
result = torch.nn.Softmax(dim=1)(pred)
print(f"result = {result}")
