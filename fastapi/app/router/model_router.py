import os

from app.gateway import model_gateway
from app.gateway.gateway_utils import MySQLConnection
from app.util import get_model_tag
from fastapi import APIRouter

router = APIRouter()


@router.get('/api/models/current')
async def current_model():
    tag = get_model_tag()
    return {'tag': tag}


@router.get('/api/models')
async def get_models():
    with MySQLConnection() as db:
        models = model_gateway.find_all(db)
        return {'models': models}
