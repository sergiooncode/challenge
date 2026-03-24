IMAGE := evaluate

.PHONY: build run test clean

build:
	docker build -t $(IMAGE) .

run: build
	docker run --env-file .env -v "$(PWD)":/app $(IMAGE)

test: build
	docker run --env-file .env -v "$(PWD)":/app $(IMAGE) pytest tests/ -v

clean:
	rm -rf output/ __pycache__ .pytest_cache
