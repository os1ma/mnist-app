import os

from app.dao import model_dao
from app.dao.dao_utils import MySQLConnection
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
        models = model_dao.find_all(db)
        return {'models': models}
