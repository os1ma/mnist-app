import os
from datetime import datetime

import onnxruntime
import torch
from fastapi import FastAPI

MODEL_FILE = '/tmp/model.onnx'

def log_info(message: str) -> None:
    now = datetime.now()
    print(f"[{now}] {message}")

app = FastAPI()

@app.get('/api/health')
async def health():
    return {'health': 'ok'}

@app.post('/api/predict')
async def predict():
    log_info("predict called.")
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)

    input_name = onnx_session.get_inputs()[0].name
    input = {input_name: torch.randn((1, 28, 28)).numpy()}
    log_info(f"input = {input}")

    output = onnx_session.run(None, input)
    log_info(f"output = {output}")

    result = output[0][0].tolist()
    log_info(f"result = {result}")

    return {'result': result}
