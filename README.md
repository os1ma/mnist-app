# mnist-app

MNIST を使った Web アプリケーションのサンプル

![mnist-app.png](./doc/mnist-app.png)

## 依存関係

- Docker
- Docker Compose
- Make

## 実行手順

### ビルド

```console
make build
```

### デフォルトのモデルでアプリケーションを起動

```console
make deploy
```

ブラウザで http://localhost:8000 にアクセスすると、推論を試すことができます

アプリケーションの停止は `make down` で可能です。

### パラメータなどを変更して推論

```console
make train
```

推論の履歴は http://localhost:5000 にアクセスすると、MLflow の UI で確認できます

※ MLflow の UI にアクセスできない場合は、`make deploy` を実行してください

`make train` での学習は CPU であり、学習で GPU を使う場合は以下のコマンドになります

```console
make train_gpu
```

### 新しいモデルをビルド・デプロイ

MLflow の UI で確認した Run の ID を指定し、モデルをビルドします

```console
make build_api RUN_ID=<Run ID>
```

対応する Docker イメージが作成されたことを、以下のコマンドで確認できます

```console
docker image ls
```

以下のコマンドにより、指定した Docker タグを持つ API をデプロイできます

```console
deploy_api RUN_ID=<RUN_ID>
```
