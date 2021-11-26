FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

COPY requirements.txt /tmp
RUN apt-get update \
  && apt-get install -y gcc libmariadb-dev \
  && pip install -r /tmp/requirements.txt

ARG MODEL_TAG=default
ENV MODEL_TAG=${MODEL_TAG}

COPY models/${MODEL_TAG}.onnx /model.onnx
