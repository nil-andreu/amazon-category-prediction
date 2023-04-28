api:
	docker build -t api-dl:v1 .
	docker run \
		-e ENV_FILE_NAME="./api/main.py" \
		-p 8080:8080 \
		-v /shared/data:/shared/data \
		api-dl:v1

train:
	docker build -t api-dl:v1 .
	docker run \
		-e ENV_FILE_NAME="./shared/model/train.py" \
		-p 8080:8080 \
		-v /shared/data:/shared/data \
		api-dl:v1

