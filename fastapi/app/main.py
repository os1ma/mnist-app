import os
from datetime import datetime

import numpy as np
import onnxruntime
from fastapi import FastAPI, File, UploadFile

MODEL_FILE = '/tmp/model.onnx'

def log_info(message: str) -> None:
    now = datetime.now()
    print(f"[{now}] {message}")

app = FastAPI()

@app.get('/api/health')
async def health():
    return {'health': 'ok'}

@app.post('/api/predict')
async def predict(image: UploadFile = File(...)):
    log_info("predict called.")

    # preprocess image file
    filename = image.filename
    log_info(f"filename = {filename}")

    # predict
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)

    random_values = np.random.random_sample((1, 28, 28))
    standardized = 2 * random_values - 1
    input = standardized.astype('float32')
    log_info(f"input.shape = {input.shape}, input[0][0][0] = {input[0][0][0]}")

    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: input})
    log_info(f"output = {output}")

    result = output[0][0].tolist()
    log_info(f"result = {result}")

    return {'result': result}
