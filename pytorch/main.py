import tempfile

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
from torchinfo import summary
from torchviz import make_dot
from tqdm import tqdm

data_root = '.'


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
    train_set0 = datasets.MNIST(
        root=data_root,
        train=True,
        download=True
    )
    mlflow.log_param("データ件数", len(train_set0))

    fig = plt.figure(figsize=(10, 3))
    for i in range(20):
        ax = plt.subplot(2, 10, i + 1)
        image, label = train_set0[i]

        plt.imshow(image, cmap='gray_r')
        ax.set_title(label)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    mlflow.log_figure(fig, 'input/data_samples.png')


def print_sample_image():
    transform1 = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set1 = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform1
    )
    image, label = train_set1[0]

    # TODO
    print(image)


def main() -> None:
    with mlflow.start_run() as run:
        save_sample_data()
        print_sample_image()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
            transforms.Lambda(lambda x: x.view(-1)),
        ])

        train_set = datasets.MNIST(
            root=data_root, train=True,
            download=True, transform=transform)
        test_set = datasets.MNIST(
            root=data_root, train=False,
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

        fig = plt.figure(figsize=(10, 3))
        for i in range(20):
            ax = plt.subplot(2, 10, i + 1)

            image = images[i].numpy()
            label = labels[i]

            image2 = (image + 1) / 2

            plt.imshow(image2.reshape(28, 28), cmap='gray_r')

            ax.set_title(f'{label}')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        mlflow.log_figure(fig, 'sample3.png')

        n_input = image.shape[0]
        n_output = len(set(list(labels.data.numpy())))
        n_hidden = 128

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
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr)

        #num_epochs = 100
        num_epochs = 3
        mlflow.log_param('num_epochs', num_epochs)

        # 評価結果記録用
        history = np.zeros((0, 5))

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
            print(
                f'Epoch [{epoch+1}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
            item = np.array(
                [epoch+1, train_loss, train_acc, val_loss, val_acc])
            history = np.vstack((history, item))

        # 損失と精度の確認
        print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}')
        print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}')

        # 学習曲線の表示 (損失)
        plt.rcParams['figure.figsize'] = (9, 8)
        fig = plt.figure()
        plt.plot(history[:, 0], history[:, 1], 'b', label='訓練')
        plt.plot(history[:, 0], history[:, 3], 'k', label='検証')
        plt.xlabel('繰り返し回数')
        plt.ylabel('損失')
        plt.title('学習曲線(損失)')
        plt.legend()
        mlflow.log_figure(fig, 'loss.png')

        # 学習曲線の表示 (精度)
        plt.rcParams['figure.figsize'] = (9, 8)
        fig = plt.figure()
        plt.plot(history[:, 0], history[:, 2], 'b', label='訓練')
        plt.plot(history[:, 0], history[:, 4], 'k', label='検証')
        plt.xlabel('繰り返し回数')
        plt.ylabel('精度')
        plt.title('学習曲線(精度)')
        plt.legend()
        mlflow.log_figure(fig, 'val.png')


if __name__ == '__main__':
    main()
