#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mep-2026-alamidrideb-bouliaire-kouka-samama
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
API_IMAGE = batchnorm-classifier-api:latest
API_PORT = 8000

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run all tests
.PHONY: test
test:
	uv run pytest tests/ -v --tb=short

## Run model tests only
.PHONY: test-models
test-models:
	uv run pytest tests/test_models.py -v --tb=short

## Run API tests only
.PHONY: test-api
test-api:
	uv run pytest tests/test_api.py -v --tb=short

## Run the API locally without Docker (development mode with auto-reload)
.PHONY: run-api
run-api:
	uv run uvicorn api.main:app --host 0.0.0.0 --port $(API_PORT) --reload

## Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(API_IMAGE) .

## Run the API with Docker Compose
.PHONY: docker-up
docker-up:
	docker compose up

## Run the API with Docker Compose in detached mode
.PHONY: docker-up-detached
docker-up-detached:
	docker compose up -d

## Stop the API
.PHONY: docker-down
docker-down:
	docker compose down

## Show API logs
.PHONY: docker-logs
docker-logs:
	docker compose logs -f api

## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	aws s3 sync s3://data-mep-project/data/ \
		data/

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	aws s3 sync data/ \
		s3://data-mep-project/data

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\.venv\Scripts\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
