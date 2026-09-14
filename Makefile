# Load .env so the CLI targets talk to the same host ports as docker compose.
# Without this, changing MLFLOW_PORT in .env would move the server but leave
# the client still pointing at the default :5000.
ifneq (,$(wildcard .env))
include .env
export
endif

# Use the local venv when present so `make test` works without activation.
PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

MODEL_NAME ?= whisper-stt
MODEL_REF  ?= production

.PHONY: up down logs ps test register-example resolve-example register-catalog catalog-dry-run clean

up:  ## bring up postgres + mlflow
	docker compose up -d --build

down:  ## stop the cluster (volumes preserved)
	docker compose down --remove-orphans

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

catalog-dry-run:  ## resolve pins + list files for every models.yaml entry, download nothing. ONLY=name to filter
	$(PYTHON) scripts/register_from_hf.py --catalog models.yaml --dry-run $(if $(ONLY),--only $(ONLY))

register-catalog:  ## import models.yaml into the registry (idempotent). ONLY=name to filter
	$(PYTHON) scripts/register_from_hf.py --catalog models.yaml $(if $(ONLY),--only $(ONLY))

clean:  ## stop the cluster AND delete all data volumes
	docker compose down -v --remove-orphans
	rm -rf cache mlruns .pytest_cache
