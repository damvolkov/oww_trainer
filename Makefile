.PHONY: setup download download-force train clean list lint type test acceptance pre-commit help

WAKEWORD      ?= $(word 2,$(MAKECMDGOALS))
N_SAMPLES     ?= 5000
N_SAMPLES_VAL ?= 1000
STEPS         ?= 10000
LANGS         ?= en

##### SETUP #####

setup: ## Install dependencies and download base assets
	@uv sync --dev
	@uv run pre-commit install
	@uv run python -m oww_trainer.download

##### DOWNLOAD #####

download: ## Download base assets only (skips already cached)
	@uv run python -m oww_trainer.download

download-force: ## Re-download all base assets
	@uv run python -m oww_trainer.download --force

##### TRAIN #####

train: ## Train a wakeword model: make train <wakeword> [LANGS=en,de,fr]
	@if [ -z "$(WAKEWORD)" ]; then \
		echo "Usage: make train <wakeword> [LANGS=en,de,fr]"; \
		echo "Example: make train eager"; \
		echo "         make train eager LANGS=en,de,fr"; \
		exit 1; \
	fi
	@uv run python -m oww_trainer.trainer "$(WAKEWORD)" \
		--n-samples $(N_SAMPLES) \
		--n-samples-val $(N_SAMPLES_VAL) \
		--steps $(STEPS) \
		--langs $(LANGS)

##### LIST #####

list: ## List all generated models (base + custom)
	@echo "=== Base models ==="
	@if [ -d models/base ]; then \
		find models/base -name '*.onnx' -o -name '*.tflite' | sort; \
	else \
		echo "  (none - run 'make setup' first)"; \
	fi
	@echo ""
	@echo "=== Custom (trained) models ==="
	@found=0; \
	for dir in models/*/; do \
		case "$$dir" in models/base/) continue ;; esac; \
		onnx=$$(find "$$dir" -name '*.onnx' 2>/dev/null); \
		if [ -n "$$onnx" ]; then \
			for f in $$onnx; do \
				size=$$(du -h "$$f" | cut -f1); \
				echo "  $$f ($$size)"; \
				found=1; \
			done; \
		fi; \
	done; \
	if [ "$$found" -eq 0 ]; then \
		echo "  (none - run 'make train <wakeword>' to create one)"; \
	fi

##### QUALITY #####

lint: ## Run ruff linter + formatter
	@uv run ruff check --fix src/ tests/
	@uv run ruff format src/ tests/

type: ## Run ty type checker
	@uv run ty check src/

test: ## Run unit tests
	@uv run pytest tests/unit/ -v

##### ACCEPTANCE #####

acceptance: ## Live wakeword test: make acceptance <model>
	@if [ -z "$(WAKEWORD)" ]; then \
		echo "Usage: make acceptance <model_name>"; \
		echo "Example: make acceptance alexa"; \
		echo "         make acceptance eager"; \
		echo ""; \
		uv run python tests/acceptance/test_wakeword.py --help; \
		exit 1; \
	fi
	@uv run python tests/acceptance/test_wakeword.py "$(WAKEWORD)"

##### CLEAN #####

clean: ## Remove generated datasets and models (keeps base)
	@echo "Removing generated datasets and models..."
	@find datasets/ -mindepth 1 -maxdepth 1 ! -name 'base' ! -name '.gitkeep' -exec rm -rf {} +
	@find models/ -mindepth 1 -maxdepth 1 ! -name 'base' ! -name '.gitkeep' -exec rm -rf {} +
	@find configs/ -name '*.yaml' -delete
	@echo "Done."

pre-commit: ## Run pre-commit on all files
	@uv run pre-commit run --all-files

##### HELP #####

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Catch-all to allow "make train eager" / "make acceptance alexa" syntax
%:
	@:
