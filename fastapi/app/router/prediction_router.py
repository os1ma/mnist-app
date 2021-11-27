import os
import uuid
from io import BytesIO

from app import predictor
from app.dao.image_dao import ImageDao
from app.dao.model_dao import ModelDao
from app.dao.prediction_dao import PredictionDao
from app.util import get_model_tag
from fastapi import APIRouter, File, UploadFile
from PIL import Image

IMAGE_DIR = '/images'

router = APIRouter()


@router.get('/api/prediction-history')
async def query_history():
    history = PredictionDao().query_history()
    return {'history': history}


@router.post('/api/predictions')
async def post_predict(image: UploadFile = File(...)):
    id = uuid.uuid4()

    # 画像を PIL のオブジェクトに変換
    pil_image = await upload_file_to_pil_image(image)
    os.makedirs(f"{IMAGE_DIR}/{id}")
    original_image_filename = f"{IMAGE_DIR}/{id}/original.png"
    pil_image.save(original_image_filename)

    # リサイズ
    resized = predictor.resize_image(pil_image)
    resized_image_filename = f"{IMAGE_DIR}/{id}/resized.png"
    resized.save(resized_image_filename)

    # 画像を保存
    image_id = ImageDao().insert(original_image_filename, resized_image_filename)

    # predict
    result = predictor.predict(resized)

    # モデルが DB に保存されていなければ保存する
    tag = get_model_tag()
    ModelDao().insert_if_not_exist(tag)
    model = ModelDao().find_by_tag(tag)
    model_id = model['id']

    # 推論結果を保存
    PredictionDao().insert(model_id, image_id, result)

    return {'result': result}


@router.post('/api/predictions/repredict-all')
async def post_predict():
    tag = get_model_tag()
    ModelDao().insert_if_not_exist(tag)
    model = ModelDao().find_by_tag(tag)

    images = ImageDao().find_all()
    for image in images:
        resized = Image.open(image['resizedFilename'])
        result = predictor.predict(resized)
        PredictionDao().insert(model['id'], image['id'], result)


async def upload_file_to_pil_image(image: UploadFile):
    data = await image.read()
    return Image.open(BytesIO(data))
