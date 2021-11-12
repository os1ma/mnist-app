import os

from fastapi import FastAPI

app = FastAPI()


@app.get('/api/health')
async def health():
    return {'health': 'ok'}

@app.post('/api/predict')
async def predict():
    return {'result': 'ok'}
