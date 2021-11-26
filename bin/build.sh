#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

readonly SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
readonly PROJECT_HOME="${SCRIPT_DIR}/.."

readonly MODEL_TAG="$1"

cp "${PROJECT_HOME}/pytorch/mlruns/0/${MODEL_TAG}/artifacts/model.onnx" \
  "${PROJECT_HOME}/fastapi/models/${MODEL_TAG}.onnx"

cd "${PROJECT_HOME}"
docker-compose build --build-arg MODEL_TAG="${MODEL_TAG}" api
