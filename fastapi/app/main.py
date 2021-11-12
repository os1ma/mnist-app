import os

import onnxruntime
import torch
from fastapi import FastAPI

MODEL_FILE = '/tmp/model.onnx'


app = FastAPI()

@app.get('/api/health')
async def health():
    return {'health': 'ok'}

@app.post('/api/predict')
async def predict():
    print("predict called.")
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)

    input_name = onnx_session.get_inputs()[0].name
    input = {input_name: torch.randn((1, 28, 28)).numpy()}
    print(f"input = {input}")

    output = onnx_session.run(None, input)
    print(f"output = {output}")

    result = output[0][0].tolist()
    print(f"result = {result}")

    return {'result': result}
