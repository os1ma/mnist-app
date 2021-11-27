import os

from app.dao.model_dao import ModelDao
from app.util import get_model_tag
from fastapi import APIRouter

router = APIRouter()


@router.get('/api/models/current')
async def current_model():
    tag = get_model_tag()
    return {'tag': tag}


@router.get('/api/models')
async def get_models():
    models = ModelDao().find_all()
    return {'models': models}
