# mnist-mlops-app

## 実行手順

### デフォルトのモデルでアプリケーションを起動

```console
docker-compose up
```

ブラウザで http://localhost:8000 にアクセスすると、推論を試すことができます

### パラメータなどを変更して推論

```console
cd pytorch
docker-compose run pytorch
```

推論の履歴は http://localhost:5000 にアクセスすると、MLflow の UI で確認できます

### 新しいモデルをビルド・デプロイ

MLflow の UI で確認した Run の ID を指定し、モデルをビルドします

```console
./bin/build.sh <Run ID>
```

```console
docker-compose up --no-deps -d api
```
