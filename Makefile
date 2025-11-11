# LLM Research Dashboard - Makefile
# Quick commands for development, testing, and deployment

.PHONY: help setup install clean test test-unit test-integration test-semantic test-quality
.PHONY: fetch analyze embed dashboard deploy
.PHONY: cost-report backup rebuild-vectors validate-quality
.PHONY: lint format check

# Default target
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "LLM Research Dashboard - Available Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup              Install dependencies, init databases (SQLite + ChromaDB)"
	@echo "  make install            Install Python dependencies only"
	@echo "  make clean              Clean cache, temp files, and Python artifacts"
	@echo ""
	@echo "Development:"
	@echo "  make dashboard          Launch Streamlit UI (localhost:8501)"
	@echo "  make deploy             Deploy to Streamlit Cloud (FREE)"
	@echo "  make fetch              Fetch papers from arXiv, X (formerly Twitter), LinkedIn"
	@echo "  make analyze            Analyze papers with LLM (uses grok-4 by default)"
	@echo "  make embed              Generate vector embeddings"
	@echo ""
	@echo "Testing:"
	@echo "  make test               Run full test suite (>80% coverage target)"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests"
	@echo "  make test-semantic      Test semantic search quality (Precision@5 >80%)"
	@echo "  make test-quality       Validate paper categorization accuracy"
	@echo ""
	@echo "Monitoring & Maintenance:"
	@echo "  make cost-report        View API spending breakdown"
	@echo "  make backup             Backup SQLite + ChromaDB"
	@echo "  make rebuild-vectors    Rebuild ChromaDB vector index"
	@echo "  make validate-quality   Run quality assurance checks"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint               Run linters (flake8, mypy)"
	@echo "  make format             Format code with black"
	@echo "  make check              Run all code quality checks"
	@echo ""

## setup: Complete setup - install dependencies and initialize databases
setup: install
	@echo "Initializing databases..."
	@mkdir -p data/chroma data/cache data/exports
	@python -c "import sqlite3; conn = sqlite3.connect('data/papers.db'); conn.executescript(open('src/storage/migrations/001_initial_schema.sql').read()); conn.close()"
	@echo "Installing Playwright browsers for LinkedIn scraping..."
	@playwright install chromium
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and add your API keys"
	@echo "  2. Run 'make test' to verify installation"
	@echo "  3. Run 'make dashboard' to launch the UI"

## install: Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed"

## clean: Clean cache, temp files, and Python artifacts
clean:
	@echo "Cleaning cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf data/cache/* 2>/dev/null || true
	@echo "✅ Cleaned"

## dashboard: Launch Streamlit dashboard
dashboard:
	@echo "Launching Streamlit dashboard on http://localhost:8501"
	@streamlit run src/dashboard/app.py

## deploy: Deploy to Streamlit Cloud (FREE tier)
deploy:
	@echo "=== Streamlit Cloud Deployment Guide ==="
	@echo ""
	@echo "1. First, push to GitHub:"
	@echo "   git add ."
	@echo "   git commit -m 'Ready for deployment'"
	@echo "   git push origin main"
	@echo ""
	@echo "2. Deploy on Streamlit Cloud:"
	@echo "   • Visit https://share.streamlit.io"
	@echo "   • Click 'New app'"
	@echo "   • Connect your GitHub repository"
	@echo "   • Main file path: src/dashboard/app.py"
	@echo ""
	@echo "3. Set environment variables in Streamlit Cloud:"
	@echo "   • XAI_API_KEY (required for LLM analysis)"
	@echo "   • OPENAI_API_KEY (required for embeddings)"
	@echo "   • TWITTER_BEARER_TOKEN (optional)"
	@echo "   • LINKEDIN_EMAIL (optional)"
	@echo ""
	@echo "✅ Your app will be live at: https://your-app.streamlit.app"
	@echo ""
	@echo "💡 Tip: Start with FREE tier, upgrade only when needed!"

## fetch: Fetch papers from all sources
fetch:
	@echo "Fetching papers from arXiv, X (formerly Twitter), and LinkedIn..."
	@python -m src.fetch.main_fetch --days 7

## analyze: Analyze papers with LLM
analyze:
	@echo "Analyzing papers with xAI grok-4-fast-reasoning..."
	@python scripts/analyze_batch.py

## embed: Generate vector embeddings
embed:
	@echo "Generating vector embeddings..."
	@python scripts/generate_embeddings.py

## test: Run full test suite
test:
	@echo "Running full test suite..."
	@pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

## test-unit: Run unit tests only
test-unit:
	@echo "Running unit tests..."
	@pytest tests/ -v -m "not integration" --cov=src --cov-report=term-missing

## test-integration: Run integration tests
test-integration:
	@echo "Running integration tests..."
	@pytest tests/test_integration.py -v

## test-semantic: Test semantic search quality
test-semantic:
	@echo "Testing semantic search quality..."
	@pytest tests/test_embeddings.py -v -k semantic

## test-quality: Validate paper categorization accuracy
test-quality:
	@echo "Validating paper categorization..."
	@python scripts/validate_quality.py --show-failures

## cost-report: View API spending breakdown
cost-report:
	@echo "Generating cost report..."
	@python scripts/cost_report.py

## backup: Backup databases
backup:
	@echo "Backing up SQLite and ChromaDB..."
	@mkdir -p data/backups
	@cp data/papers.db data/backups/papers_$(shell date +%Y%m%d_%H%M%S).db
	@tar -czf data/backups/chroma_$(shell date +%Y%m%d_%H%M%S).tar.gz data/chroma/
	@echo "✅ Backup complete in data/backups/"

## rebuild-vectors: Rebuild ChromaDB vector index
rebuild-vectors:
	@echo "Rebuilding ChromaDB vector index..."
	@python scripts/rebuild_vector_index.py

## validate-quality: Run quality assurance checks
validate-quality:
	@echo "Running quality assurance checks..."
	@python scripts/validate_quality.py

## lint: Run linters
lint:
	@echo "Running flake8..."
	@flake8 src/ tests/ --max-line-length=100 --exclude=__pycache__
	@echo "Running mypy..."
	@mypy src/ --ignore-missing-imports

## format: Format code with black
format:
	@echo "Formatting code with black..."
	@black src/ tests/ --line-length=100

## check: Run all code quality checks
check: lint
	@echo "Running tests..."
	@pytest tests/ -v -x
	@echo "✅ All checks passed"
