.ONESHELL:
API_PORT ?= 5000
all: train

env:
	mamba env update -n dev -f environment.yml

train:
	rm -Rf mlflow-hello/saved_model
	mlflow run mlflow-hello

test_saved_model:
	mlflow run mlflow-hello -e test_saved_model

serve:
	mlflow run mlflow-hello -e serve

serve_wrong_model_fp:
	mlflow run mlflow-hello -e serve -P model_fp=noway

test_api:
	mlflow run mlflow-hello -e test_api -P api_port=$(API_PORT)

build_docker:
	cd mlflow-hello && docker build . -t mlflow-hello

serve_docker:
	docker run --rm -it -p "$(API_PORT):5000" mlflow-hello
