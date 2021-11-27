from fastapi import FastAPI

from app.router import model_router, prediction_router

app = FastAPI()
app.include_router(model_router.router)
app.include_router(prediction_router.router)


@app.get('/api/health')
async def health():
    return {'health': 'ok'}
