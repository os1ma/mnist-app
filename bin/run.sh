#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

readonly SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
readonly PROJECT_HOME="${SCRIPT_DIR}/.."

cd "${PROJECT_HOME}/pytorch"
docker-compose down
docker-compose up --build pytorch

cp "${PROJECT_HOME}/pytorch/model.onnx" "${PROJECT_HOME}/fastapi/model.onnx"

cd "${PROJECT_HOME}"
docker-compose down
docker-compose up --build -d
