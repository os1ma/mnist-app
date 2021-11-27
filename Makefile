PROJECT_HOME := $(shell pwd)

.PHONY: build
build:
	docker-compose build \
	&& cd pytorch \
	&& docker-compose build

.PHONY: build_api
build_api:
	cp "${PROJECT_HOME}/pytorch/mlruns/0/${RUN_ID}/artifacts/model.onnx" \
	"${PROJECT_HOME}/fastapi/models/${RUN_ID}.onnx" \
	&& API_DOCKER_TAG="${RUN_ID}" docker-compose build --build-arg MODEL_TAG="${RUN_ID}" api

.PHONY: deploy
deploy:
	docker-compose up -d

.PHONY: deploy_api
deploy_api:
	API_DOCKER_TAG="${RUN_ID}" docker-compose up --no-deps -d api

.PHONY: train
train:
	cd pytorch \
	&& docker-compose run pytorch

.PHONY: train_gpu
train_gpu:
	cd pytorch \
	&& docker-compose -f docker-compose.yaml -f docker-compose.override-gpu.yaml run pytorch

.PHONY: connect_db
connect_db:
	docker-compose exec db mysql -uapp -ppassword app

.PHONY: down
down:
	docker-compose down
