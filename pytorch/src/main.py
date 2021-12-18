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


class MNISTNet(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

        # TODO Flatten と Sequential を使う
        self.l1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3


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

        save_sample_data()

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
        test_set = datasets.MNIST(
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
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False
        )

        for X, y in train_loader:
            break

        n_input = X[0].shape[0]
        n_output = len(set(list(y.data.numpy())))
        n_hidden = 3
        mlflow.log_param('n_input', n_input)
        mlflow.log_param('n_output', n_output)
        mlflow.log_param('n_hidden', n_hidden)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.log_param('device', device)

        seed = 123
        mlflow.log_param('seed', seed)
        fix_seed(seed)

        lr = 0.01
        mlflow.log_param('lr', lr)

        net = MNISTNet(n_input, n_output, n_hidden).to(device)
        logger.info(f"net = {net}")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)

        epochs = 3
        mlflow.log_param('num_epochs', epochs)

        for t in range(epochs):
            epoch = t+1
            mlflow.log_metric('epoch', epoch)

            # 訓練フェーズ
            train_acc_sum, train_loss_sum = 0, 0
            for X, y in tqdm(train_loader):
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                outputs = net(X)

                loss = loss_fn(outputs, y)
                loss.backward()

                optimizer.step()

                predicted = torch.max(outputs, 1)[1]

                train_loss_sum += loss.item()
                train_acc_sum += (predicted == y).sum()

            # 評価値の算出・記録
            train_loss = train_loss_sum * batch_size / len(train_set)
            train_acc = train_acc_sum / len(train_set)
            mlflow.log_metric('train_loss', train_loss, epoch)
            mlflow.log_metric('train_acc', train_acc.item(), epoch)

            # 予測フェーズ
            val_acc_sum, val_loss_sum = 0, 0
            for X, y in test_loader:
                X = X.to(device)
                y = y.to(device)

                outputs_test = net(X)

                loss = loss_fn(outputs_test, y)

                predicted_test = torch.max(outputs_test, 1)[1]

                val_loss_sum += loss.item()
                val_acc_sum += (predicted_test == y).sum()

            # 評価値の算出・記録
            val_loss = val_loss_sum * batch_size / len(test_set)
            val_acc = val_acc_sum / len(test_set)
            mlflow.log_metric('val_loss', val_loss, epoch)
            mlflow.log_metric('val_acc', val_acc.item(), epoch)

            logger.info(
                f'Epoch [{epoch}/{epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')

        save_model_as_onnx(net)


if __name__ == '__main__':
    main()
