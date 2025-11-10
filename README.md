# LLM Research Dashboard

Automated system for discovering, analyzing, and organizing LLM research papers using AI, organized by the 2025 Smol Training Playbook pipeline stages with semantic search capabilities.

## Overview

**Purpose:** Help researchers stay current with LLM research by automatically categorizing papers into pipeline stages, providing semantic search, and tracking professional/social impact.

**Key Features:**
- 📚 **Multi-source data fetching:** arXiv, X (formerly Twitter), LinkedIn
- 🤖 **AI-powered analysis:** xAI grok-4-fast-reasoning for paper categorization
- 🔍 **Semantic search:** OpenAI embeddings + ChromaDB vector database
- 📊 **8 Pipeline stages:** Architecture, Data Prep, Pre-Training, Post-Training, Evaluation, Infrastructure, Deployment, Emerging
- 💰 **Cost-optimized:** $13-20/month for 1000 papers/day

## Tech Stack

- **Python 3.11+**
- **LLM:** xAI grok-4-fast-reasoning (primary), Together AI (fallbacks)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** ChromaDB
- **Database:** SQLite
- **Dashboard:** Streamlit
- **Data Sources:** arXiv API, X API (formerly Twitter), LinkedIn

## Project Status

**Current Phase:** Planning Complete → Phase 1 (Foundation & Setup) starting

See [PROJECT_PLAN.md](./PROJECT_PLAN.md) for complete 6-week implementation plan.

## Quick Start

```bash
# Coming soon - Phase 1 in progress
make setup      # Install dependencies, init databases
make fetch      # Fetch papers from sources
make analyze    # Analyze papers with LLM
make dashboard  # Launch Streamlit UI
```

## Documentation

- **[PROJECT_PLAN.md](./PROJECT_PLAN.md)** - Complete 6-week implementation plan with architecture, phases, and specifications
- **[CLAUDE.md](./CLAUDE.md)** - AI assistant guidance for Claude Code sessions

## Development Principles

- **Test-Driven Development:** Write tests before implementation
- **Cost tracking:** Every API call logged and monitored
- **Incremental commits:** Small, tested features
- **Configuration over code:** Use YAML configs, not hardcoded values

## License

MIT License - See LICENSE file for details

## Author

John Dvorak (john.n.dvorak@gmail.com)

---

**Timeline:** 6 weeks from start to production
**Budget:** $13-20/month operating costs
**Last Updated:** 2025-11-08
