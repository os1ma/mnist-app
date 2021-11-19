from datetime import datetime
from io import BytesIO

import numpy as np
import onnxruntime
from fastapi import FastAPI, File, UploadFile
from PIL import Image

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
    data = await image.read()
    pil_image = Image.open(BytesIO(data))
    # 28 * 28 に変換
    resized = pil_image.resize((28, 28))
    arr = np.array(resized)
    log_info(f"arr.shpae = {arr.shape}")
    # 軸の変換
    transposed = arr.transpose()[3:]
    log_info(f"transposed.shape = {transposed.shape}, transposed = {transposed}, max = {np.amax(transposed)}")
    # 型を変換
    typed = transposed.astype('float32')
    # 0 ~ 1 に正規化
    standardized = typed / 255
    log_info(f"standardized.shape = {standardized.shape}, standardized = {standardized}")

    # predict
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: standardized})
    log_info(f"output = {output}")

    result = output[0][0].tolist()
    log_info(f"result = {result}")

    return {'result': result}
