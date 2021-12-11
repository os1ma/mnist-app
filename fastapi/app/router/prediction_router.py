import os
import uuid
from io import BytesIO

import app.stream_logger as stream_logger
from app import predictor
from app.config import IMAGE_DIR, MODEL_TAG
from app.gateway import image_gateway, model_gateway, prediction_gateway
from app.gateway.mysql_connection import MySQLConnection
from fastapi import APIRouter, File, UploadFile
from PIL import Image

logger = stream_logger.of(__name__)

router = APIRouter()


@router.get('/api/prediction-history')
async def query_history():
    with MySQLConnection() as db:
        history = prediction_gateway.query_history(db)
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
        image_id = image_gateway.insert(
            db, original_image_filename, resized_image_filename)

        # predict
        result = predictor.predict(resized)

        # モデルが DB に保存されていなければ保存する
        model_gateway.insert_if_not_exist(db, MODEL_TAG)
        model = model_gateway.find_by_tag(db, MODEL_TAG)
        model_id = model['id']

        # 推論結果を保存
        prediction_gateway.insert(db, model_id, image_id, result)

        db.commit()

        return {'result': result}


@router.post('/api/predictions/repredict-all')
async def post_predict():
    with MySQLConnection() as db:
        model_gateway.insert_if_not_exist(db, MODEL_TAG)
        model = model_gateway.find_by_tag(db, MODEL_TAG)
        logger.info(f"model = {model}")

        images = image_gateway.find_all(db)
        for image in images:
            resized = Image.open(image['resizedFilename'])
            result = predictor.predict(resized)
            prediction_gateway.insert(db, model['id'], image['id'], result)

        db.commit()


async def upload_file_to_pil_image(image: UploadFile):
    data = await image.read()
    return Image.open(BytesIO(data))
