FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

RUN apt-get update \
  && apt-get install -y gcc libmariadb-dev \
  && pip install mysqlclient
