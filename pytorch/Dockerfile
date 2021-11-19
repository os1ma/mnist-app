FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

COPY requirements.txt /tmp

RUN apt-get update \
  && apt-get install -y python3 python3-pip \
  && pip install -r /tmp/requirements.txt
