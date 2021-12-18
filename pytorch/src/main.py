import tempfile
from datetime import datetime

import japanize_matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import stream_logger

DATA_ROOT = '.'
MODEL_OUTPUT_FILE = './model.onnx'

logger = stream_logger.of(__name__)


class MNISTModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


def save_sample_data():
    train_set = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True
    )
    mlflow.log_param("データ件数", len(train_set))

    fig = plt.figure(figsize=(10, 3))
    for i in range(20):
        ax = plt.subplot(2, 10, i + 1)
        image, label = train_set[i]

        plt.imshow(image, cmap='gray_r')
        ax.set_title(label)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    mlflow.log_figure(fig, 'input/data_samples.png')


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def train(dataloader, model, loss_fn, optimizer, device):
    loss_sum = 0
    acc_sum = 0

    for X, y in tqdm(dataloader):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = loss_fn(outputs, y)
        loss.backward()

        optimizer.step()

        loss_sum += loss.item()

        predicted = torch.max(outputs, 1)[1]
        acc_sum += (predicted == y).sum()

    return loss_sum, acc_sum


def validation(dataloader, model, loss_fn, device):
    loss_sum = 0
    acc_sum = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        outputs = model(X)

        loss = loss_fn(outputs, y)

        loss_sum += loss.item()

        predicted = torch.max(outputs, 1)[1]
        acc_sum += (predicted == y).sum()

    return loss_sum, acc_sum


def save_model_as_onnx(net):
    dummy_input = torch.randn((1, 28, 28)).view(-1)
    net.cpu()
    logger.info("export onnx model")
    torch.onnx.export(net,
                      dummy_input,
                      MODEL_OUTPUT_FILE,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'])
    mlflow.log_artifact(MODEL_OUTPUT_FILE)


def main() -> None:
    with mlflow.start_run():
        mlflow.log_artifact('./src/')

        seed = 123
        mlflow.log_param('seed', seed)
        fix_seed(seed)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.log_param('device', device)

        save_sample_data()

        # データの準備

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ])

        train_set = datasets.MNIST(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=transform
        )
        valid_set = datasets.MNIST(
            root=DATA_ROOT,
            train=False,
            download=True,
            transform=transform
        )

        batch_size = 64
        mlflow.log_param('batch_size', batch_size)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False
        )

        # モデルの準備

        for X, y in train_loader:
            break

        n_input = X[0].shape[0]
        n_output = len(set(list(y.data.numpy())))
        n_hidden = 2
        mlflow.log_param('n_input', n_input)
        mlflow.log_param('n_output', n_output)
        mlflow.log_param('n_hidden', n_hidden)

        model = MNISTModel(n_input, n_output, n_hidden).to(device)
        logger.info(f"model = {model}")
        loss_fn = nn.CrossEntropyLoss()

        lr = 0.01
        mlflow.log_param('lr', lr)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # エポック数だけ学習・評価を繰り返す

        epochs = 5
        mlflow.log_param('num_epochs', epochs)

        for t in range(epochs):
            epoch = t+1
            mlflow.log_metric('epoch', epoch)

            # 学習

            train_loss_sum, train_acc_sum = train(
                train_loader, model, loss_fn, optimizer, device)

            train_loss = train_loss_sum * batch_size / len(train_set)
            train_acc = train_acc_sum / len(train_set)
            mlflow.log_metric('train_loss', train_loss, epoch)
            mlflow.log_metric('train_acc', train_acc.item(), epoch)
            logger.info(
                f'Epoch [{epoch}/{epochs}] train loss: {train_loss:.5f} acc: {train_acc:.5f}')

            # 評価

            valid_loss_sum, valid_acc_sum = validation(
                valid_loader, model, loss_fn, device)

            valid_loss = valid_loss_sum * batch_size / len(valid_set)
            valid_acc = valid_acc_sum / len(valid_set)
            mlflow.log_metric('valid_loss', valid_loss, epoch)
            mlflow.log_metric('valid_acc', valid_acc.item(), epoch)
            logger.info(
                f'Epoch [{epoch}/{epochs}] valid loss: {valid_loss:.5f}, acc: {valid_acc:.5f}')

        # モデルを保存

        save_model_as_onnx(model)


if __name__ == '__main__':
    main()
