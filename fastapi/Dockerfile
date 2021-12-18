FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

COPY requirements.txt /requirements.txt
RUN apt-get update \
  && apt-get install -y gcc libmariadb-dev \
  && pip install -r /requirements.txt

ARG MODEL_TAG=v1
ENV MODEL_TAG=${MODEL_TAG}

COPY models/${MODEL_TAG}.onnx /model.onnx
