import os

from app.config import MODEL_TAG
from app.gateway import model_gateway
from app.gateway.mysql_connection import MySQLConnection
from fastapi import APIRouter

router = APIRouter()


@router.get('/api/models/current')
async def current_model():
    return {'tag': MODEL_TAG}


@router.get('/api/models')
async def get_models():
    with MySQLConnection() as db:
        models = model_gateway.find_all(db)
        return {'models': models}
