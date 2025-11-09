# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Research Dashboard** - Automated system for discovering, analyzing, and organizing LLM research papers with semantic search.

**IMPORTANT:** Full implementation plan is in `PROJECT_PLAN.md`. Always read that file first for architecture, phases, and detailed specifications.

**Tech Stack:** Python 3.11+ | xAI grok-4-fast-reasoning | OpenAI embeddings | ChromaDB | Streamlit | SQLite

**Cost:** $13-20/month for 1000 papers/day

## Commands

### Setup & Development
```bash
make setup              # Install dependencies, init databases (SQLite + ChromaDB)
make fetch              # Fetch papers from arXiv, Twitter, LinkedIn
make analyze            # Analyze papers with LLM (uses grok-4 by default)
make embed              # Generate vector embeddings
make dashboard          # Launch Streamlit UI (localhost:8501)
```

### Testing
```bash
make test               # Run full test suite (target: >80% coverage)
make test-unit          # Unit tests only
make test-semantic      # Test semantic search quality (Precision@5 >80%)
```

### Monitoring
```bash
make cost-report        # View API spending breakdown
make backup             # Backup SQLite + ChromaDB
```

## Code Style

**IMPORTANT: Follow these conventions strictly**

- Use ES modules syntax for any JS: `import`/`export`, not `require`
- Python type hints everywhere: `def func(x: str) -> Dict[str, Any]:`
- Docstrings for all public functions with examples
- Configuration over hardcoding - use YAML configs
- Error handling with structured logging (loguru)

**Example:**
```python
from src.utils.logger import logger

def analyze_paper(abstract: str, title: str) -> Dict[str, Any]:
    """
    Analyze paper and categorize into pipeline stages.

    Args:
        abstract: Paper abstract text
        title: Paper title

    Returns:
        Dict with stages, summary, key_insights
    """
    try:
        result = llm_provider.analyze(abstract, title)
        logger.info(f"Analyzed: {title}", extra={"stages": result['stages']})
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {e}", extra={"title": title})
        raise
```

## Development Workflow

**CRITICAL: Test-Driven Development**

1. Read `PROJECT_PLAN.md` for current phase tasks
2. Write tests BEFORE implementation
3. Implement feature
4. Run tests: `make test`
5. Commit incrementally with clear messages
6. Update docs if needed

**NEVER:**
- ❌ Skip writing tests for "simple" code
- ❌ Hardcode API keys (use .env)
- ❌ Re-analyze/re-embed papers without checking cache first
- ❌ Commit large files (use .gitignore for data/)

## Key Files & Purposes

**Configuration (edit these to change behavior):**
- `config/stages.yaml` - 8 pipeline stages with keywords
- `config/llm_config.yaml` - LLM provider settings, costs, fallback rules
- `config/queries.yaml` - Search queries for arXiv/Twitter/LinkedIn

**Core Modules (most frequently modified):**
- `src/llm/provider_factory.py` - Provider selection logic
- `src/analysis/prompts.py` - LLM prompts for categorization
- `src/embeddings/semantic_search.py` - Semantic search implementation
- `src/fetch/linkedin_fetcher.py` - LinkedIn integration

**Database:**
- `src/storage/paper_db.py` - SQLite CRUD operations (skeleton created)
- `src/storage/migrations/001_initial_schema.sql` - Complete schema (implemented ✅)
- See `PROJECT_PLAN.md` for schema (tables: papers, linkedin_posts, cost_tracking)
- **NOTE:** LinkedIn fields included in initial schema (no 002 migration needed)

## Testing Strategy

**Unit Tests:** Mock all external APIs (arXiv, Twitter, LinkedIn, LLM providers)
**Integration Tests:** End-to-end: Fetch → Analyze → Embed → Store → Query
**Quality Tests:** Manually labeled papers, validate >90% accuracy

**Test fixtures:** `tests/fixtures/sample_papers.json`

## Multi-LLM Provider System

**PRIMARY:** xAI grok-4-fast-reasoning (95% of papers) - $0.20 input / $0.50 output per 1M tokens

**Fallbacks (automatic):**
- Together AI GLM-4.6 - If rate limit hit
- Together AI DeepSeek-V3 - If API errors
- Together AI Qwen3-Thinking - Top 5% complexity only

**IMPORTANT:** Cost tracking is mandatory. Every API call must be logged to `cost_tracking` table.

**To add a new provider:**
1. Implement `LLMProvider` interface in `src/llm/providers/`
2. Add to `config/llm_config.yaml`
3. Update `provider_factory.py` selection logic
4. Test with `pytest tests/test_llm_providers.py::test_new_provider`

## Vector Embeddings

**Provider:** OpenAI text-embedding-3-small (1536 dims, $0.02/1M tokens)
**Storage:** ChromaDB collection "llm_papers" at `data/chroma/`
**Input format:** `{title} [SEP] {abstract} [SEP] {key_insights}`
**Batch size:** 100 papers per API call

**IMPORTANT:** Always check if embedding exists before regenerating (costly)

## LinkedIn Integration

**CRITICAL:** LinkedIn scraping requires careful rate limiting to avoid blocks

**Rate limits:**
- 5 seconds delay between requests
- Max 100 posts per day
- Rotate user agents

**If blocked:**
1. Increase delay: Edit `config/queries.yaml` → `rate_limit_delay: 10`
2. Reduce volume: `max_posts_per_day: 50`
3. Use LinkedIn API instead of scraping (requires developer account)

**Tracked companies:** OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft Research, Hugging Face

## Common Tasks

### Improving Stage Categorization Accuracy

If accuracy drops below 90%:
1. `python scripts/validate_quality.py --show-failures`
2. Edit `src/analysis/prompts.py` - add few-shot examples
3. Temporarily use better model: Set `primary_provider: "together"` and `primary_model: "qwen3-thinking"` in config
4. Re-validate: `make test-quality`

### Debugging Semantic Search

If search returns irrelevant results:
1. Check ChromaDB: `collection.count()` should match papers in DB
2. Test similarity: Values should be 0.0-1.0
3. Try different embedding model: Edit `config/embedding_config.yaml`
4. Rebuild index: `make rebuild-vectors`

### Cost Overruns

If daily costs exceed budget:
1. Check: `make cost-report`
2. Auto-fallback should trigger to cheaper provider
3. Manually switch: `make analyze PROVIDER=together MODEL=glm`
4. Reduce volume or increase budget in `config/llm_config.yaml`

## Project-Specific Quirks

**Database:**
- SQLite timeout set to 30s to avoid lock errors
- ChromaDB requires `PersistentClient(path="data/chroma")` for persistence

**API Providers:**
- xAI uses OpenAI-compatible client (base_url="https://api.x.ai/v1")
- Together AI has 3 models - selection based on paper complexity
- LinkedIn API unofficial library can break on UI changes

**Embeddings:**
- Generating 1000 embeddings takes ~8 minutes (batched)
- ChromaDB similarity search should be <100ms

**Data Sources:**
- arXiv rate limit: 1 request per 3 seconds (strict)
- Twitter free tier: 10k tweets/month
- LinkedIn scraping: Watch for CAPTCHA or login prompts

## Environment Variables

**Required:**
```bash
XAI_API_KEY=xai-xxx              # Primary LLM
OPENAI_API_KEY=sk-xxx            # Embeddings
TWITTER_BEARER_TOKEN=AAA...      # Social metrics
LINKEDIN_EMAIL=your@email.com    # Professional metrics
LINKEDIN_PASSWORD=yourpass
```

**Optional fallbacks:**
```bash
TOGETHER_API_KEY=xxx
GOOGLE_API_KEY=xxx
ANTHROPIC_API_KEY=xxx
```

## Success Criteria

Before moving to next phase:
- [ ] All tests passing (>80% coverage)
- [ ] Stage categorization >90% accuracy
- [ ] Semantic search Precision@5 >80%
- [ ] Cost tracking working
- [ ] No API keys in code
- [ ] Performance benchmarks met (see PROJECT_PLAN.md)

## Resources

**Project Docs:**
- `PROJECT_PLAN.md` - Complete 6-week implementation plan (READ THIS FIRST)
- `WORKFLOW.md` - Git workflow and best practices

**External:**
- xAI API: https://docs.x.ai/
- ChromaDB: https://docs.trychroma.com/
- 2025 Smol Training Playbook (Hugging Face)

---

**Last Updated:** 2025-11-09
**Current Phase:** Phase 1 (Foundation & Setup) - Steps 1-3 Complete ✅
**Current Status:** Ready for Step 4 (Database Implementation)

**Next Tasks (Step 4):**
- Implement SQLite database with ORM (paper_db.py)
- Implement ChromaDB vector store (vector_store.py)
- Create database migration runner
- Write comprehensive database tests

**Steps 1-3 Completion Summary:**

**Step 1: Project Structure (Commit: 9b74bc8)**
- ✅ Complete project structure (36 files, 547 lines)
- ✅ All Python modules with type hints & docstrings
- ✅ Database schema with LinkedIn & embedding fields (001_initial_schema.sql)
- ✅ Test structure with TDD-ready fixtures

**Step 2: Development Environment (Commit: 070510a)**
- ✅ requirements.txt with 33 dependencies
- ✅ Makefile with 15 commands (setup, test, fetch, analyze, dashboard)
- ✅ .env.example with all API keys and configuration options
- ✅ Environment setup validated with pytest

**Step 3: Configuration System (Commit: 24cf4fe)**
- ✅ 5 YAML configuration files (stages.yaml, llm_config.yaml, embedding_config.yaml, queries.yaml, budget_modes.yaml)
- ✅ config_loader.py with 7 helper functions
- ✅ 200+ stage keywords, 6 LLM providers, 3 budget modes configured
- ✅ 10/10 tests passing, TDD workflow validated

**Comprehensive Unit Testing (Post Steps 1-3):**
- ✅ 6 new test files: test_utils.py, test_llm_providers.py, test_fetchers.py, test_analysis.py, test_storage.py, test_embeddings.py
- ✅ 213 new tests covering all modules from Steps 1-3
- ✅ 71% overall code coverage (100% for all implemented modules)
- ✅ All 223 tests passing (213 new + 10 config tests)
- ✅ Test execution: <1 second (excellent performance)
- ✅ Test categories: interface contracts, parameter validation, integration scenarios, edge cases, unicode handling

**Total Progress:** 52 files, 3,955 lines of code (2,072 source + 1,883 test code)
