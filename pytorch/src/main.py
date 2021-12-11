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

logger = stream_logger.of(__name__)


class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()

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
            root=DATA_ROOT, train=True,
            download=True, transform=transform)
        test_set = datasets.MNIST(
            root=DATA_ROOT, train=False,
            download=True, transform=transform)

        batch_size = 50
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

        for images, labels in train_loader:
            break

        n_input = images[0].shape[0]
        n_output = len(set(list(labels.data.numpy())))
        n_hidden = 3
        mlflow.log_param('n_input', n_input)
        mlflow.log_param('n_output', n_output)
        mlflow.log_param('n_hidden', n_hidden)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mlflow.log_param('device', device)

        seed = 123
        mlflow.log_param('seed', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True

        lr = 0.01
        mlflow.log_param('lr', lr)

        net = Net(n_input, n_output, n_hidden).to(device)
        logger.info(f"net = {net}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)

        num_epochs = 3
        mlflow.log_param('num_epochs', num_epochs)

        for epoch in range(num_epochs):
            train_acc, train_loss = 0, 0
            val_acc, val_loss = 0, 0
            n_train, n_test = 0, 0

            # 訓練フェーズ
            for inputs, labels in tqdm(train_loader):
                n_train += len(labels)

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                predicted = torch.max(outputs, 1)[1]

                train_loss += loss.item()
                train_acc += (predicted == labels).sum()

            # 予測フェーズ
            for inputs_test, labels_test in test_loader:
                n_test += len(labels_test)

                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(inputs_test)

                loss_test = criterion(outputs_test, labels_test)

                predicted_test = torch.max(outputs_test, 1)[1]

                val_loss += loss_test.item()
                val_acc += (predicted_test == labels_test).sum()

            # 評価値の算出・記録
            train_acc = train_acc / n_train
            val_acc = val_acc / n_test
            train_loss = train_loss * batch_size / n_train
            val_loss = val_loss * batch_size / n_test

            outputEpoch = epoch+1
            logger.info(
                f'Epoch [{outputEpoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
            mlflow.log_metric('epoch', outputEpoch)
            mlflow.log_metric('train_loss', train_loss, outputEpoch)
            mlflow.log_metric('train_acc', train_acc.item(), outputEpoch)
            mlflow.log_metric('val_loss', val_loss, outputEpoch)
            mlflow.log_metric('val_acc', val_acc.item(), outputEpoch)

        filepath = './model.onnx'
        dummy_input = torch.randn((1, 28, 28)).view(-1)
        net.cpu()
        logger.info("export onnx model")
        torch.onnx.export(net, dummy_input, filepath, verbose=True,
                          input_names=['input'], output_names=['output'])
        mlflow.log_artifact(filepath)


if __name__ == '__main__':
    main()
