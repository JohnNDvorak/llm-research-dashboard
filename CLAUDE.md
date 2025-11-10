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
make fetch              # Fetch papers from arXiv, X (formerly Twitter), LinkedIn
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
- `config/queries.yaml` - Search queries for arXiv/X/LinkedIn

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

**Unit Tests:** Mock all external APIs (arXiv, X, LinkedIn, LLM providers)
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

**CRITICAL:** LinkedIn scraping requires careful rate limiting and anti-detection measures

**Dual-Mode Operation:**
- **Primary:** Web scraping with Playwright (more flexible)
- **Fallback:** LinkedIn API (if credentials available)
- **Automatic:** Switch between modes based on availability/blocking

**Rate Limiting (Conservative):**
- Base delay: 5 seconds between requests (with ±2s jitter)
- Max 100 posts/day (configurable in `config/queries.yaml`)
- Session rotation: Every 50 posts
- Daily pause: After 80 posts to reduce detection risk
- User agent rotation: 5 different browser signatures

**Anti-Detection Measures:**
- Human-like behavior simulation (scrolls, delays, mouse movements)
- CAPTCHA detection and pause on detection
- Proxy rotation support (configure if needed)
- Session persistence and rotation
- Block detection with automatic mode switching

**If Blocked:**
1. Automatic fallback to API mode (if credentials available)
2. Increase delay: Edit `config/queries.yaml` → `rate_limit_delay: 10`
3. Reduce volume: `max_posts_per_day: 50`
4. Enable proxy rotation
5. Switch to API-only mode (requires developer account)

**Professional Score Calculation:**
```python
professional_score = (likes * 1) + (comments * 5) + (shares * 3) + (views * 0.001)
# Apply 1.5x multiplier for verified researchers/companies
```

**Tracked Companies:**
OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft Research, Hugging Face, Cohere, Inflection AI

**Implementation Pattern:**
```python
from src.fetch.linkedin_fetcher import LinkedinFetcher

fetcher = LinkedinFetcher()
papers = fetcher.fetch_recent_papers(days=7)
# Returns standardized format for PaperDeduplicator
```

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
- X free tier: 500k posts/month
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

**Last Updated:** 2025-11-10
**Current Phase:** Phase 2 (Paper Fetching) - IN PROGRESS 🚧 (75% Complete)
**Current Status:** Phase 2.3 (X Fetcher) - COMPLETE with 100% test coverage

**Session 2025-11-10 Summary:**
- ✅ Implemented Paper Deduplicator (Phase 2.2) - 515 lines, 45/45 tests passing
- ✅ Validated Phase 1+2 Integration - 12/12 integration tests passing
- ✅ Cleaned up test suite - 315/315 tests passing (100% pass rate)
- ✅ Fixed 3 integration issues (schema, serialization, indexing)
- ✅ Enhanced Phase 2.3 (X Fetcher) plan with detailed specifications
- ✅ Updated PROJECT_PLAN.md v1.1 → v1.2 with X implementation details
- ✅ **Implemented X Fetcher (440 lines) + comprehensive tests (634 lines)**
- ✅ **Updated entire codebase from Twitter to X branding**
- ✅ **Achieved perfect test coverage: 338/338 tests passing (100%)**
- ✅ **Fixed all 10 mock test configuration issues**
- ✅ **Committed all changes to git (commit 46fa1aa)**

**Current Phase 2 Progress:**
- ✅ ArXiv Fetcher (Phase 2.1): COMPLETE - 33/33 tests passing (100%)
- ✅ Paper Deduplicator (Phase 2.2): COMPLETE - 45/45 tests passing (100%)
- ✅ Phase 1+2 Integration: VALIDATED - 12/12 integration tests passing
- ✅ **X Fetcher (Phase 2.3): COMPLETE** - 440 lines implementation, 22 tests, 100% coverage
- ✅ **X/Twitter Branding: COMPLETE** - All references updated, backward compatible
- 📋 LinkedIn Fetcher (Phase 2.4): TODO - Most complex, web scraping

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

**Step 4: Database Implementation (Commit: d8b0f12)**
- ✅ SQLite database: src/storage/paper_db.py (445 lines)
  - Full CRUD operations with JSON serialization
  - Migration system: execute_migration()
  - Filtering, pagination, cost tracking
  - Context manager support
- ✅ ChromaDB vector store: src/embeddings/vector_store.py (388 lines)
  - Persistent vector storage with PersistentClient
  - Batch operations, similarity search with filtering
  - Full CRUD for embeddings
  - Context manager support
- ✅ Integration tests: test_database_integration.py (455 lines, 16 tests)
  - All 16 tests passing (100%)
  - Tests cover SQLite, ChromaDB, and cross-database workflows

**Total Progress (Phase 1 Complete):** 58 files, 6,741 lines of code (3,319 source + 3,422 test code)

**Step 5 Complete (2025-11-09):**
- ✅ Implemented src/utils/logger.py (119 lines)
  - 3 loguru handlers: console (colorized), file (rotation), error-only
  - Compression, async logging, dynamic level changes
- ✅ Logging directory auto-creation on import

**Step 6 Complete (2025-11-09):**
- ✅ Verified `make setup` command completes successfully
  - Dependencies installed, databases initialized, Playwright browsers installed
- ✅ Created tests/test_logger.py (295 lines, 29 tests)
  - All 29 tests passing in 1.24s
  - Comprehensive coverage: configuration, integration, edge cases

**🎉 PHASE 1 COMPLETE - All 6 Steps Finished 🎉**
- All tests passing (258/258, 100%) after fixing test compatibility
- Production-ready logging infrastructure
- Database systems fully implemented and tested
- All tests fixed and validated

**🚀 PHASE 2 IN PROGRESS - Paper Fetching**
- ArXiv fetcher complete (420 lines, 32/34 tests passing)
- Rate limiting: 3 seconds between requests
- Database integration confirmed
- Ready for next fetchers
