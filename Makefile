# Use the local venv when present so `make test` works without activation.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

MODEL_NAME ?= whisper-stt
MODEL_REF  ?= production

.PHONY: up down logs ps test register-example resolve-example clean

up:  ## bring up postgres + minio + mlflow
	docker compose up -d --build

down:  ## stop the cluster (volumes preserved)
	docker compose down

logs:  ## follow the mlflow server log
	docker compose logs -f mlflow

ps:  ## show service status
	docker compose ps

test:  ## run client tests (sqlite, no docker needed)
	$(PYTHON) -m pytest tests/ -q

register-example:  ## register examples/fake_model and promote it to @production
	$(PYTHON) scripts/register_pretrained.py \
		--name $(MODEL_NAME) \
		--source-dir examples/fake_model \
		--alias $(MODEL_REF) \
		--desc "example pretrained model" \
		--tag framework=whisper \
		--tag registered_by=makefile

resolve-example:  ## resolve whatever @production currently points at
	$(PYTHON) scripts/registry_cli.py resolve \
		--name $(MODEL_NAME) --ref $(MODEL_REF) --cache-dir ./cache

clean:  ## stop the cluster AND delete all data volumes
	docker compose down -v
	rm -rf cache mlruns .pytest_cache
