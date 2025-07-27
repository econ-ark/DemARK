# DemARK Makefile
# Provides automation for common development tasks

.PHONY: help lint-md fix-md test-md-env clean

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Markdown linting targets
lint-md: ## Lint all markdown files
	@echo "🔍 Linting all markdown files..."
	@python tools/lint_markdown.py $$(find . -name "*.md" -not -path "./.git/*" -not -path "./.conda_env/*" -not -path "./project/repos/*")

fix-md: ## Fix all markdown files automatically
	@echo "🔧 Fixing all markdown files..."
	@python tools/lint_markdown.py $$(find . -name "*.md" -not -path "./.git/*" -not -path "./.conda_env/*" -not -path "./project/repos/*") --fix

test-md-env: ## Test markdown linting environment
	@echo "🧪 Testing markdown linting environment..."
	@python tools/lint_markdown.py --test-env

# Development targets
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.log" -delete
	@find . -type f -name "jupyter_start_*.log" -delete

# Documentation targets
docs: ## Build documentation
	@echo "📚 Building documentation..."
	@jupyter-book build .

docs-clean: ## Clean documentation build
	@echo "🧹 Cleaning documentation build..."
	@rm -rf _build/

# Testing targets
test: ## Run all tests
	@echo "🧪 Running tests..."
	@python -m pytest --nbval-lax --nbval-cell-timeout=12000 --ignore=notebooks/Chinese-Growth.ipynb --ignore=notebooks/Harmenberg-Aggregation.ipynb notebooks/

# Quality assurance
qa: lint-md test ## Run quality assurance checks
	@echo "✅ Quality assurance complete" 