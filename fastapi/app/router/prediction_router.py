import os
import uuid
from io import BytesIO

from app import predictor
from app.dao import image_dao, model_dao, prediction_dao
from app.dao.dao_utils import MySQLConnection
from app.util import get_model_tag, log_info
from fastapi import APIRouter, File, UploadFile
from PIL import Image

IMAGE_DIR = '/images'

router = APIRouter()


@router.get('/api/prediction-history')
async def query_history():
    with MySQLConnection() as db:
        history = prediction_dao.query_history(db)
        return {'history': history}


@router.post('/api/predictions')
async def post_predict(image: UploadFile = File(...)):
    with MySQLConnection() as db:
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
        image_id = image_dao.insert(
            db, original_image_filename, resized_image_filename)

        # predict
        result = predictor.predict(resized)

        # モデルが DB に保存されていなければ保存する
        tag = get_model_tag()
        model_dao.insert_if_not_exist(db, tag)
        model = model_dao.find_by_tag(db, tag)
        model_id = model['id']

        # 推論結果を保存
        prediction_dao.insert(db, model_id, image_id, result)

        db.commit()

        return {'result': result}


@router.post('/api/predictions/repredict-all')
async def post_predict():
    with MySQLConnection() as db:
        tag = get_model_tag()
        model_dao.insert_if_not_exist(db, tag)
        model = model_dao.find_by_tag(db, tag)
        log_info(f"model = {model}")

        images = image_dao.find_all(db)
        for image in images:
            resized = Image.open(image['resizedFilename'])
            result = predictor.predict(resized)
            prediction_dao.insert(db, model['id'], image['id'], result)

        db.commit()


async def upload_file_to_pil_image(image: UploadFile):
    data = await image.read()
    return Image.open(BytesIO(data))
