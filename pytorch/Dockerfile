FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt /

# git は MLflow が内部で利用するため
RUN apt-get update \
  && apt-get install -y python3 python3-pip git graphviz \
  && pip install -r /requirements.txt
