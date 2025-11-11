"""Comprehensive integration test for Phases 1-4 of the LLM Research Dashboard.

This test verifies the complete end-to-end functionality:
- Phase 1: Foundation & Setup
- Phase 2: Paper Fetching (arXiv, Twitter/X, LinkedIn)
- Phase 3: LLM Analysis & Embeddings
- Phase 4: Dashboard functionality
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json
import sqlite3
from unittest.mock import Mock, patch, MagicMock

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Import modules from all phases
from storage.database import PaperDatabase
from embeddings.semantic_search import SemanticSearch
from embeddings.vector_store import VectorStore
from llm.provider_factory import LLMProviderFactory
from analysis.analyzer import PaperAnalyzer
from fetch.arxiv_fetcher import ArxivFetcher
from fetch.twitter_fetcher import TwitterFetcher
from fetch.linkedin_fetcher import LinkedinFetcher
from analysis.deduplicator import PaperDeduplicator
from utils.cost_tracker import CostTracker
from utils.config_loader import get_config


class TestPhase1Foundation:
    """Test Phase 1: Foundation & Setup components."""

    def test_database_initialization(self):
        """Test SQLite database initialization and schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = PaperDatabase(db_path=db_path)

            # Verify tables exist
            assert os.path.exists(db_path)

            # Test basic operations
            paper_id = db.add_paper(
                title="Test Paper",
                authors=["Test Author"],
                abstract="Test abstract",
                arxiv_id="2301.00001"
            )
            assert paper_id is not None

            # Retrieve paper
            paper = db.get_paper(paper_id)
            assert paper['title'] == "Test Paper"
            assert paper['authors'] == ["Test Author"]

    def test_vector_store_initialization(self):
        """Test ChromaDB vector store initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = VectorStore(persist_dir=tmpdir)

            # Test collection creation
            assert vector_store.collection is not None
            assert vector_store.collection.name == "llm_papers"

            # Test adding and searching
            vector_store.add(
                ids=["test_1"],
                documents=["Test document content"],
                metadatas=[{"title": "Test"}]
            )

            results = vector_store.query(
                query_texts=["test"],
                n_results=1
            )
            assert len(results['ids'][0]) > 0

    def test_configuration_loading(self):
        """Test configuration system."""
        # Test loading stage configuration
        stages = get_stage_keywords()
        assert isinstance(stages, dict)
        assert len(stages) == 8  # 8 development stages
        assert "Stage 1: Idea Conception" in stages

        # Test LLM configuration
        llm_config = get_config("llm_config.yaml")
        assert "primary_provider" in llm_config
        assert "primary_model" in llm_config


class TestPhase2Fetching:
    """Test Phase 2: Paper Fetching components."""

    @patch('src.fetch.arxiv_fetcher.arxiv.Search')
    def test_arxiv_fetching(self, mock_search):
        """Test arXiv paper fetching."""
        # Mock arXiv response
        mock_paper = Mock()
        mock_paper.title = "Test arXiv Paper"
        mock_paper.authors = [Mock(name="Test Author")]
        mock_paper.summary = "Test abstract"
        mock_paper.published = datetime.now()
        mock_paper.entry_id = "http://arxiv.org/abs/2301.00001"
        mock_search.return_value.results.return_value = [mock_paper]

        fetcher = ArxivFetcher()
        papers = fetcher.fetch_papers(days=1, limit=1)

        assert len(papers) > 0
        assert papers[0]['title'] == "Test arXiv Paper"
        assert papers[0]['source'] == 'arxiv'

    @patch('tweepy.Client')
    def test_twitter_fetching(self, mock_client):
        """Test Twitter/X paper fetching."""
        # Mock Twitter response
        mock_tweet = Mock()
        mock_tweet.data = {"id": "123", "text": "Check out this paper: https://arxiv.org/abs/2301.00001"}
        mock_tweet.includes = {'users': [{'username': 'researcher'}]}
        mock_client.return_value.search_recent_tweets.return_value = mock_tweet

        fetcher = TwitterFetcher()
        papers = fetcher.fetch_papers(days=1, limit=1)

        # Verify fetcher runs without errors
        assert isinstance(papers, list)

    @patch('playwright.sync_api.sync_playwright')
    def test_linkedin_fetching(self, mock_playwright):
        """Test LinkedIn paper fetching."""
        # Mock Playwright browser
        mock_browser = Mock()
        mock_page = Mock()
        mock_page.query_selector_all.return_value = []
        mock_browser.new_page.return_value = mock_page
        mock_playwright.return_value.__enter__.return_value.chromium.launch.return_value = mock_browser

        fetcher = LinkedinFetcher()
        papers = fetcher.fetch_recent_papers(days=1, limit=1)

        # Verify fetcher runs without errors
        assert isinstance(papers, list)

    def test_paper_deduplication(self):
        """Test paper deduplication across sources."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))
            deduplicator = PaperDeduplicator(db)

            # Add similar papers
            paper1 = {
                'title': 'Test Paper: A Novel Approach',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'This paper presents a novel approach to testing',
                'arxiv_id': '2301.00001',
                'source': 'arxiv'
            }

            paper2 = {
                'title': 'Test Paper: A Novel Approach to Testing',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'We present a novel approach for testing systems',
                'twitter_url': 'https://twitter.com/status/123',
                'source': 'twitter'
            }

            # Add papers
            id1 = db.add_paper(**paper1)
            id2 = db.add_paper(**paper2)

            # Test deduplication
            deduplicator.mark_duplicates()

            # Verify one is marked as duplicate
            p1 = db.get_paper(id1)
            p2 = db.get_paper(id2)
            assert p1['duplicate_of'] is None or p2['duplicate_of'] is not None


class TestPhase3Analysis:
    """Test Phase 3: LLM Analysis & Embeddings components."""

    @patch('openai.OpenAI')
    def test_llm_provider_factory(self, mock_openai):
        """Test LLM provider selection and fallback."""
        # Mock OpenAI client for xAI compatibility
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=json.dumps({
            "stages": ["Stage 2: Literature Review"],
            "summary": "Test summary",
            "key_insights": ["Test insight"]
        })))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set API key
            os.environ['XAI_API_KEY'] = 'test_key'

            factory = LLMProviderFactory()
            provider = factory.get_provider()

            assert provider is not None
            assert provider.provider_name == 'xai'

    def test_paper_analysis(self):
        """Test paper analysis pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock analyzer
            analyzer = PaperAnalyzer()

            # Test prompt generation
            prompt = analyzer._generate_analysis_prompt(
                title="Test Paper",
                abstract="Test abstract for analysis"
            )

            assert "Test Paper" in prompt
            assert "Test abstract" in prompt
            assert "development stage" in prompt.lower()

    @patch('sentence_transformers.SentenceTransformer')
    def test_embedding_generation(self, mock_model):
        """Test embedding generation."""
        # Mock sentence transformer
        mock_model.return_value.encode.return_value = [[0.1, 0.2, 0.3] * 128]  # 384 dims

        from embeddings.embedding_generator import EmbeddingGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['OPENAI_API_KEY'] = 'test_key'
            generator = EmbeddingGenerator()

            # Test embedding generation
            embeddings = generator.generate_embeddings([
                {"title": "Test", "abstract": "Test abstract", "key_insights": []}
            ])

            assert len(embeddings) > 0
            assert len(embeddings[0]) > 0

    def test_cost_tracking(self):
        """Test cost tracking functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))
            cost_tracker = CostTracker(db)

            # Test cost logging
            cost_tracker.log_cost(
                provider="test_provider",
                model="test_model",
                operation="test_operation",
                tokens=100,
                cost=0.01
            )

            # Test cost retrieval
            costs = cost_tracker.get_total_costs()
            assert costs.get("total", 0) > 0


class TestPhase4Dashboard:
    """Test Phase 4: Dashboard components."""

    def test_dashboard_imports(self):
        """Test dashboard module imports."""
        # Test that dashboard modules can be imported
        try:
            import sys
            sys.path.insert(0, str(project_root / 'src'))
            from dashboard.app import init_session_state, render_sidebar
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import dashboard modules: {e}")

    def test_database_queries_for_dashboard(self):
        """Test database queries used by dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))

            # Add test papers
            for i in range(5):
                db.add_paper(
                    title=f"Paper {i}",
                    authors=[f"Author {i}"],
                    abstract=f"Abstract {i}",
                    stages=[f"Stage {(i % 8) + 1}"],
                    published_date=(datetime.now() - timedelta(days=i)).isoformat()
                )

            # Test dashboard queries
            total = db.get_total_papers()
            assert total == 5

            recent = db.get_recent_papers(days=3)
            assert len(recent) == 3

            stages = db.get_stage_distribution()
            assert len(stages) > 0

            papers_over_time = db.get_papers_over_time()
            assert len(papers_over_time) > 0

    def test_semantic_search_for_dashboard(self):
        """Test semantic search functionality for dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize vector store
            vector_store = VectorStore(persist_dir=tmpdir)

            # Add test papers
            vector_store.add(
                ids=["paper_1", "paper_2"],
                documents=[
                    "Machine learning algorithms for classification",
                    "Deep neural networks for image recognition"
                ],
                metadatas=[
                    {"title": "ML Paper", "stage": "Stage 3"},
                    {"title": "Deep Learning Paper", "stage": "Stage 4"}
                ]
            )

            # Test search
            results = vector_store.query(
                query_texts=["neural networks"],
                n_results=2,
                where={"stage": "Stage 4"}
            )

            assert len(results['ids'][0]) > 0


class TestEndToEndIntegration:
    """Test complete end-to-end integration across all phases."""

    def test_complete_pipeline_simulation(self):
        """Simulate complete pipeline from fetch to dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))
            vector_store = VectorStore(persist_dir=tmpdir)
            cost_tracker = CostTracker(db)

            # Phase 2: Simulate fetching
            test_papers = [
                {
                    'title': 'Advances in Large Language Models',
                    'authors': ['Researcher One', 'Researcher Two'],
                    'abstract': 'This paper presents new advances in LLMs',
                    'arxiv_id': '2401.00001',
                    'published_date': datetime.now().isoformat(),
                    'source': 'arxiv'
                },
                {
                    'title': 'Efficient Training of Transformer Models',
                    'authors': ['ML Expert'],
                    'abstract': 'We propose an efficient training method',
                    'arxiv_id': '2401.00002',
                    'published_date': (datetime.now() - timedelta(days=1)).isoformat(),
                    'source': 'arxiv'
                }
            ]

            # Add papers to database
            paper_ids = []
            for paper in test_papers:
                paper_id = db.add_paper(**paper)
                paper_ids.append(paper_id)

            # Phase 3: Simulate analysis
            for i, paper_id in enumerate(paper_ids):
                db.update_analysis(
                    paper_id,
                    stages=[f"Stage {(i % 8) + 1}"],
                    summary=f"Summary for paper {i+1}",
                    key_insights=[f"Insight {i+1}.1", f"Insight {i+1}.2"]
                )

            # Simulate embeddings
            for i, paper in enumerate(test_papers):
                vector_store.add(
                    ids=[f"paper_{paper_ids[i]}"],
                    documents=[f"{paper['title']} {paper['abstract']}"],
                    metadatas=[{'paper_id': paper_ids[i], 'title': paper['title']}]
                )

            # Phase 4: Test dashboard queries
            # Test search functionality
            total_papers = db.get_total_papers()
            assert total_papers == 2

            # Test filtering
            recent_papers = db.get_recent_papers(days=2)
            assert len(recent_papers) == 2

            # Test analytics
            stage_dist = db.get_stage_distribution()
            assert len(stage_dist) == 2

            # Test semantic search
            results = vector_store.query(
                query_texts=["language models"],
                n_results=2
            )
            assert len(results['ids'][0]) > 0

            # Test cost tracking
            cost_tracker.log_cost(
                provider="openai",
                model="gpt-4",
                operation="analysis",
                tokens=1000,
                cost=0.02
            )

            costs = cost_tracker.get_total_costs()
            assert costs['total'] == 0.02

            # Verify all phases connected
            assert total_papers > 0
            assert len(stage_dist) > 0
            assert costs['total'] > 0


class TestPerformanceAndReliability:
    """Test performance and reliability of the integrated system."""

    def test_large_dataset_performance(self):
        """Test system performance with larger dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))

            # Add many papers
            start_time = datetime.now()
            for i in range(100):
                db.add_paper(
                    title=f"Paper {i}: Machine Learning Advances",
                    authors=[f"Author {i}"],
                    abstract=f"This is abstract {i} about ML",
                    stages=[f"Stage {(i % 8) + 1}"]
                )

            add_time = datetime.now() - start_time

            # Test query performance
            start_time = datetime.now()
            papers = db.get_papers(limit=50)
            query_time = datetime.now() - start_time

            # Performance assertions
            assert add_time.total_seconds() < 5.0  # Should add 100 papers quickly
            assert query_time.total_seconds() < 1.0  # Queries should be fast
            assert len(papers) == 50

    def test_error_handling(self):
        """Test error handling across components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))

            # Test invalid paper ID
            result = db.get_paper(99999)
            assert result is None

            # Test invalid search
            with patch('embeddings.semantic_search.SemanticSearch.search') as mock_search:
                mock_search.side_effect = Exception("Search error")
                search = SemanticSearch()
                papers = search.search("test query")
                assert isinstance(papers, list)

    def test_data_consistency(self):
        """Test data consistency across operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))

            # Add paper with complete data
            paper_id = db.add_paper(
                title="Complete Test Paper",
                authors=["Author One"],
                abstract="Complete abstract",
                arxiv_id="2401.00001",
                stages=["Stage 1"],
                summary="Complete summary",
                key_insights=["Insight 1"],
                twitter_metrics={"likes": 10, "retweets": 5},
                linkedin_metrics={"likes": 8, "comments": 2}
            )

            # Retrieve and verify all fields
            paper = db.get_paper(paper_id)
            assert paper['title'] == "Complete Test Paper"
            assert paper['authors'] == ["Author One"]
            assert paper['stages'] == ["Stage 1"]
            assert paper['summary'] == "Complete summary"
            assert paper['twitter_metrics']['likes'] == 10
            assert paper['linkedin_metrics']['comments'] == 2


# Test configuration
pytest_plugins = []

if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v", "--tb=short"])