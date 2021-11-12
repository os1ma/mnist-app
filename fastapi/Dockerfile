FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

COPY model.onnx /tmp
COPY requirements.txt /tmp

RUN pip install -r /tmp/requirements.txt
