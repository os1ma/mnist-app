import os
import uuid
from datetime import datetime
from io import BytesIO

import numpy as np
import onnxruntime
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from scipy.special import softmax

from app.dao.image_dao import ImageDao
from app.dao.model_dao import ModelDao
from app.dao.prediction_dao import PredictionDao

MODEL_FILE = '/model.onnx'
IMAGE_DIR = '/images'


def log_info(message: str) -> None:
    now = datetime.now()
    print(f"[{now}] {message}")


def get_model_tag() -> str:
    return os.environ['MODEL_TAG']


app = FastAPI()


@app.get('/api/health')
async def health():
    return {'health': 'ok'}


@app.get('/api/models/current')
async def current_model():
    tag = get_model_tag()
    return {'tag': tag}


@app.get('/api/models')
async def get_models():
    models = ModelDao().find_all()
    return {'models': models}


@app.get('/api/prediction-history')
async def query_history():
    history = PredictionDao().query_history()
    return {'history': history}


def predict(resized_image):
    arr = np.array(resized_image)
    # 軸の変換
    transposed = arr.transpose()[3:]
    log_info(
        f"transposed.shape = {transposed.shape}, transposed = {transposed}, max = {np.amax(transposed)}")
    # 型を変換
    typed = transposed.astype('float32')
    # 0 ~ 1 に正規化
    standardized = typed / 255
    log_info(
        f"standardized.shape = {standardized.shape}, standardized = {standardized}")

    # predict
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: standardized})
    log_info(f"output = {output}")

    result = softmax(output[0][0]).tolist()
    log_info(f"result = {result}")

    return result


@app.post('/api/predictions')
async def post_predict(image: UploadFile = File(...)):
    id = uuid.uuid4()
    log_info(f"predict called. id = {id}")

    # preprocess image file

    filename = image.filename
    log_info(f"filename = {filename}")
    data = await image.read()
    pil_image = Image.open(BytesIO(data))
    os.makedirs(f"{IMAGE_DIR}/{id}")
    original_image_filename = f"{IMAGE_DIR}/{id}/original.png"
    pil_image.save(original_image_filename)
    # 28 * 28 に変換
    resized = pil_image.resize((28, 28))
    resized_image_filename = f"{IMAGE_DIR}/{id}/resized.png"
    resized.save(resized_image_filename)

    # 画像を保存
    image_id = ImageDao().insert(original_image_filename, resized_image_filename)
    log_info(f"image_id = {image_id}")

    # predict
    result = predict(resized)

    # モデルが DB に保存されていなければ保存する
    tag = get_model_tag()
    ModelDao().insert_if_not_exist(tag)
    model = ModelDao().find_by_tag(tag)
    model_id = model['id']
    log_info(f"model_id = {model_id}")

    # 推論結果を保存
    PredictionDao().insert(model_id, image_id, result)

    return {'result': result}


@app.post('/api/predictions/repredict-all')
async def post_predict():
    tag = get_model_tag()
    model = ModelDao().find_by_tag(tag)

    images = ImageDao().find_all()
    for image in images:
        resized = Image.open(image['resizedFilename'])
        result = predict(resized)
        PredictionDao().insert(model['id'], image['id'], result)
