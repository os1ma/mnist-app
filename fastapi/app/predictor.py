import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from scipy.special import softmax

from app.util import log_info

MODEL_FILE = '/model.onnx'


def resize_image(original_image):
    return original_image.resize((28, 28))


def predict(resized_image):
    input = _preprocess(resized_image)

    # predict
    onnx_session = onnxruntime.InferenceSession(MODEL_FILE)
    input_name = onnx_session.get_inputs()[0].name
    output = onnx_session.run(None, {input_name: input})
    log_info(f"output = {output}")

    # 確率に変換
    result = softmax(output[0]).tolist()
    log_info(f"result = {result}")

    return result


def _preprocess(resized_image):
    arr = np.array(resized_image)
    log_info(f"arr.shape = {arr.shape}")

    # 軸の変換
    transposed = arr.transpose()[3:]
    log_info(
        f"transposed.shape = {transposed.shape}, transposed = {transposed}, max = {np.amax(transposed)}")

    # 学習時と同様の変換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.Lambda(lambda x: x.view(-1)),
    ])
    input = transform(transposed).numpy()
    log_info(
        f"input.shape = {input.shape}, input = {input}, max = {np.amax(input)}")

    return input
