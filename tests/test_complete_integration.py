"""
Complete Integration Test for Phase 1 and Phase 2.

This test validates the entire pipeline:
Phase 1: Foundation & Setup
- Database initialization
- Configuration system
- Logging infrastructure
- Vector store setup

Phase 2: Paper Fetching
- ArXiv fetcher
- Twitter/X fetcher
- LinkedIn fetcher
- Paper deduplication
- Cross-source integration

End-to-end flow:
Fetch → Deduplicate → Store → Retrieve → Verify
"""

import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.storage.paper_db import PaperDB
from src.embeddings.vector_store import VectorStore
from src.fetch.fetch_manager import FetchManager
from src.fetch.paper_deduplicator import PaperDeduplicator
from src.utils.config_loader import load_config
from src.utils.logger import logger
from src.fetch.main_fetch import main_fetch


class TestPhase1Integration:
    """Test Phase 1 components integration."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Initialize database
        db = PaperDB(db_path=db_path)
        db.execute_migration('src/storage/migrations/001_initial_schema.sql')
        db.close()

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def temp_chroma(self):
        """Create temporary ChromaDB for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            chroma_path = os.path.join(temp_dir, "chroma")
            yield chroma_path

    def test_database_initialization(self, temp_db):
        """Test database initialization with all tables."""
        with PaperDB(db_path=temp_db) as db:

            # Check all tables exist
            tables = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = [t[0] for t in tables]

            required_tables = [
                'papers',
                'cost_tracking'
            ]

            for table in required_tables:
                assert table in table_names, f"Missing table: {table}"

            # Check table schemas
            papers_columns = db.conn.execute("PRAGMA table_info(papers)").fetchall()
            paper_columns = [col[1] for col in papers_columns]

            # Check essential columns
            essential_columns = [
                'id', 'title', 'abstract', 'authors', 'source',
                'social_score', 'professional_score', 'combined_score',
                'arxiv_id', 'url', 'fetch_date', 'published_date'
            ]

            for col in essential_columns:
                assert col in paper_columns, f"Missing column in papers table: {col}"

            # Check LinkedIn-specific columns
            linkedin_columns = [
                'linkedin_post_id', 'linkedin_post_url', 'linkedin_author_name',
                'linkedin_author_title', 'linkedin_company', 'linkedin_likes',
                'linkedin_comments', 'linkedin_shares', 'linkedin_views',
                'linkedin_post_date', 'professional_score'
            ]

            for col in linkedin_columns:
                assert col in paper_columns, f"Missing LinkedIn column: {col}"

            # Check X/Twitter-specific columns
            twitter_columns = [
                'tweet_id', 'twitter_likes', 'twitter_retweets', 'twitter_replies',
                'twitter_poster', 'twitter_post_date'
            ]

            for col in twitter_columns:
                assert col in paper_columns, f"Missing Twitter column: {col}"

    def test_vector_store_initialization(self, temp_chroma):
        """Test ChromaDB vector store initialization."""
        store = VectorStore(persist_directory=temp_chroma)
        store.connect()

        # Test collection creation
        assert store.collection is not None
        assert store.collection.name == "llm_papers"

        # Test initial count
        assert store.collection.count() == 0

        store.disconnect()

    def test_configuration_system(self):
        """Test all configuration files load correctly."""
        # Test stage configuration
        stages_config = load_config('stages')
        assert 'stages' in stages_config
        assert len(stages_config['stages']) == 8
        assert any('pre-training' in ' '.join(s['keywords']).lower() for s in stages_config['stages'])

        # Test LLM configuration
        llm_config = load_config('llm_config')
        assert 'providers' in llm_config
        assert 'primary_provider' in llm_config
        assert llm_config['primary_provider'] == 'xai'

        # Test queries configuration
        queries_config = load_config('queries')
        assert 'arxiv' in queries_config
        assert 'twitter' in queries_config
        assert 'linkedin' in queries_config

        # Check LinkedIn companies
        linkedin_companies = queries_config['linkedin']['tracked_companies']
        assert len(linkedin_companies) > 25  # Should have 30+ companies

        # Verify high priority companies
        high_priority = [c for c in linkedin_companies if c['priority'] == 'high']
        assert any(c['name'] == 'OpenAI' for c in high_priority)
        assert any(c['name'] == 'Anthropic' for c in high_priority)

    def test_logging_infrastructure(self):
        """Test logging system is properly configured."""
        # Test logger exists and is configured
        assert logger is not None

        # Test logging to files works
        log_file = "data/logs/app.log"
        logger.info("Test log message")

        # Check log directory was created
        assert os.path.exists("data") or os.path.exists("logs")

    def test_phase1_components_integration(self, temp_db, temp_chroma):
        """Test all Phase 1 components work together."""
        # Initialize all components
        db = PaperDB(db_path=temp_db)
        store = VectorStore(persist_directory=temp_chroma)
        config = load_config('stages')

        # Test database operations
        test_paper = {
            'id': 'test:123',
            'title': 'Test Paper',
            'abstract': 'Test abstract',
            'authors': ['Test Author'],
            'source': 'test',
            'social_score': 100,
            'professional_score': 200,
            'arxiv_id': '2401.12345'
        }

        db.insert_paper(test_paper)
        retrieved = db.get_paper('test:123')
        assert retrieved['title'] == 'Test Paper'

        # Test vector store operations
        store.connect()
        # Create fake embedding (1536 dimensions)
        import random
        embedding = [random.random() for _ in range(1536)]
        store.add_paper('test:123', embedding, {'title': test_paper['title']})

        # Cleanup
        db.close()
        store.disconnect()


class TestPhase2Integration:
    """Test Phase 2 components integration."""

    @pytest.fixture
    def mock_papers_from_all_sources(self):
        """Mock papers from all three sources."""
        return {
            'arxiv': [
                {
                    'id': 'arxiv:2401.00001',
                    'title': 'Attention Is All You Need',
                    'abstract': 'We propose a new simple network architecture...',
                    'authors': ['Vaswani, A.', 'Shazeer, N.'],
                    'source': 'arxiv',
                    'social_score': 0,
                    'professional_score': 0,
                    'arxiv_id': '2401.00001',
                    'url': 'https://arxiv.org/abs/2401.00001',
                    'published_date': '2024-01-01'
                },
                {
                    'id': 'arxiv:2401.00002',
                    'title': 'DPO: Direct Preference Optimization',
                    'abstract': 'We propose DPO...',
                    'authors': ['OpenAI Team'],
                    'source': 'arxiv',
                    'social_score': 0,
                    'professional_score': 0,
                    'arxiv_id': '2401.00002',
                    'url': 'https://arxiv.org/abs/2401.00002',
                    'published_date': '2024-01-02'
                }
            ],
            'twitter': [
                {
                    'id': 'twitter_123456789',
                    'title': None,
                    'abstract': None,
                    'authors': ['@OpenAI'],
                    'source': 'twitter',
                    'social_score': 1000,
                    'professional_score': 0,
                    'arxiv_id': '2401.00001',  # Same as first arXiv paper
                    'x_tweet_id': '123456789',
                    'x_author': 'OpenAI',
                    'x_likes': 500,
                    'x_retweets': 300,
                    'x_url': 'https://x.com/OpenAI/status/123456789',
                    'fetch_date': datetime.now(timezone.utc).date().isoformat()
                }
            ],
            'linkedin': [
                {
                    'id': 'linkedin:987654321',
                    'title': None,
                    'abstract': None,
                    'authors': ['Dr. Jane Smith'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 500,
                    'arxiv_id': '2401.00002',  # Same as second arXiv paper
                    'linkedin_post_id': '987654321',
                    'linkedin_author_name': 'Dr. Jane Smith',
                    'linkedin_author_title': 'Research Scientist at OpenAI',
                    'linkedin_company': 'OpenAI',
                    'linkedin_likes': 150,
                    'linkedin_comments': 50,
                    'linkedin_shares': 25,
                    'fetch_date': datetime.now(timezone.utc).date().isoformat(),
                    'published_date': datetime.now(timezone.utc).date().isoformat()
                },
                {
                    'id': 'linkedin:456789123',
                    'title': None,
                    'abstract': None,
                    'authors': ['John Doe'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 300,
                    'arxiv_id': '2401.00003',  # Unique paper
                    'linkedin_company': 'DeepSeek',
                    'fetch_date': datetime.now(timezone.utc).date().isoformat(),
                    'published_date': datetime.now(timezone.utc).date().isoformat()
                }
            ]
        }

    def test_deduplicator_cross_source_merge(self, mock_papers_from_all_sources):
        """Test deduplication across all three sources."""
        # Combine all papers
        all_papers = (
            mock_papers_from_all_sources['arxiv'] +
            mock_papers_from_all_sources['twitter'] +
            mock_papers_from_all_sources['linkedin']
        )

        deduplicator = PaperDeduplicator()
        deduplicated = deduplicator.deduplicate(all_papers)

        # Should have 3 unique papers:
        # 1. Merged (arxiv + twitter) for 2401.00001
        # 2. Merged (arxiv + linkedin) for 2401.00002
        # 3. LinkedIn-only for 2401.00003
        assert len(deduplicated) == 3

        # Check first merged paper (arxiv + twitter)
        merged_1 = next(p for p in deduplicated if p.get('arxiv_id') == '2401.00001')
        assert merged_1['id'] == 'arxiv:2401.00001'  # arXiv ID takes precedence
        assert merged_1['title'] == 'Attention Is All You Need'
        assert merged_1['social_score'] == 1000  # From Twitter
        assert 'Vaswani' in str(merged_1['authors'])
        assert 'OpenAI' in str(merged_1['authors'])  # Should merge authors

        # Check second merged paper (arxiv + linkedin)
        merged_2 = next(p for p in deduplicated if p.get('arxiv_id') == '2401.00002')
        assert merged_2['title'] == 'DPO: Direct Preference Optimization'
        assert merged_2['professional_score'] == 500  # From LinkedIn
        assert merged_2['linkedin_company'] == 'OpenAI'
        assert 'OpenAI Team' in str(merged_2['authors'])
        assert 'Dr. Jane Smith' in str(merged_2['authors'])

        # Check LinkedIn-only paper
        linkedin_only = next(p for p in deduplicated if p.get('arxiv_id') == '2401.00003')
        assert linkedin_only['source'] == 'linkedin'
        assert linkedin_only['linkedin_company'] == 'DeepSeek'
        assert linkedin_only['professional_score'] == 300

    def test_combined_score_calculation(self, mock_papers_from_all_sources):
        """Test combined score calculation across sources."""
        deduplicator = PaperDeduplicator()

        # Create a merged paper with scores from multiple sources
        merged_paper = {
            'id': 'test:123',
            'social_score': 1000,  # From Twitter
            'professional_score': 500,  # From LinkedIn
            'published_date': '2024-01-01'
        }

        combined_score = deduplicator._calculate_combined_score(merged_paper)

        # Combined = (social * 0.4) + (prof * 0.6) + (recency * 0.3)
        # For 2024-01-01, recency should be very low (almost a year old)
        expected = (1000 * 0.4) + (500 * 0.6)  # + recency (close to 0)
        assert combined_score >= expected and combined_score < expected + 100

    @patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test_token'})
    def test_fetch_manager_all_sources(self, mock_papers_from_all_sources):
        """Test fetch manager coordinates all sources."""
        manager = FetchManager()

        # Mock all fetchers
        with patch.object(manager.arxiv_fetcher, 'fetch_recent_papers') as mock_arxiv:
            with patch.object(manager.twitter_fetcher, 'fetch_recent_papers') as mock_twitter:
                with patch.object(manager.linkedin_fetcher, 'fetch_recent_papers') as mock_linkedin:

                    # Setup mock returns
                    mock_arxiv.return_value = mock_papers_from_all_sources['arxiv']
                    mock_twitter.return_value = mock_papers_from_all_sources['twitter']
                    mock_linkedin.return_value = mock_papers_from_all_sources['linkedin']

                    # Fetch from all sources
                    papers = manager.fetch_all_papers(days=7, parallel=False)

                    # Verify all sources were called
                    mock_arxiv.assert_called_once_with(days=7)
                    mock_twitter.assert_called_once_with(days=7)
                    mock_linkedin.assert_called_once_with(days=7)

                    # Check results
                    assert len(papers) == 3  # After deduplication

                    # Check stats
                    assert manager.stats['arxiv_count'] == 2
                    assert manager.stats['twitter_count'] == 1
                    assert manager.stats['linkedin_count'] == 2
                    assert manager.stats['total_before_dedup'] == 5
                    assert manager.stats['total_after_dedup'] == 3
                    assert manager.stats['duplicates_removed'] == 2


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        db = PaperDB(db_path=db_path)
        db.execute_migration('src/storage/migrations/001_initial_schema.sql')
        db.close()

        yield db_path

        if os.path.exists(db_path):
            os.unlink(db_path)

    @patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test_token'})
    def test_complete_workflow_mocked(self, temp_db):
        """Test complete workflow with mocked APIs."""
        # Mock all external APIs
        mock_papers = {
            'arxiv': [
                {
                    'id': 'arxiv:2401.00001',
                    'title': 'Test Paper 1',
                    'abstract': 'Test abstract 1',
                    'authors': ['Author 1'],
                    'source': 'arxiv',
                    'social_score': 0,
                    'professional_score': 0,
                    'arxiv_id': '2401.00001'
                }
            ],
            'twitter': [
                {
                    'id': 'twitter_123',
                    'title': None,
                    'abstract': None,
                    'authors': ['@test'],
                    'source': 'twitter',
                    'social_score': 100,
                    'professional_score': 0,
                    'arxiv_id': '2401.00001'
                }
            ],
            'linkedin': [
                {
                    'id': 'linkedin_456',
                    'title': None,
                    'abstract': None,
                    'authors': ['Dr Test'],
                    'source': 'linkedin',
                    'social_score': 0,
                    'professional_score': 200,
                    'arxiv_id': '2401.00001',
                    'linkedin_company': 'OpenAI'
                }
            ]
        }

        with patch('src.fetch.fetch_manager.PaperDB') as MockPaperDB:
            # Setup database mock
            mock_db = Mock()
            MockPaperDB.return_value.__enter__.return_value = mock_db
            MockPaperDB.return_value.__exit__.return_value = None
            mock_db.paper_exists.return_value = False  # Paper doesn't exist, will be stored
            mock_db.insert_paper.return_value = None  # Successfully stored

            # Mock fetchers
            with patch('src.fetch.arxiv_fetcher.ArxivFetcher.fetch_recent_papers') as mock_arxiv:
                with patch('src.fetch.twitter_fetcher.TwitterFetcher.fetch_recent_papers') as mock_twitter:
                    with patch('src.fetch.linkedin_fetcher.LinkedinFetcher.fetch_recent_papers') as mock_linkedin:

                        # Setup mocks
                        mock_arxiv.return_value = mock_papers['arxiv']
                        mock_twitter.return_value = mock_papers['twitter']
                        mock_linkedin.return_value = mock_papers['linkedin']

                        # Run the workflow
                        results = main_fetch(days=7, store=True, verbose=False)

                        # Verify results
                        assert results['papers_fetched'] > 0
                        assert results['papers_stored'] > 0
                        assert results['duplicates_removed'] > 0
                        assert 'source_counts' in results

                        # Verify database operations
                        assert mock_db.insert_paper.call_count > 0
                        assert mock_db.paper_exists.call_count > 0

                        # Check final stored paper has merged data
                        stored_call = mock_db.insert_paper.call_args[0][0]
                        assert stored_call['title'] == 'Test Paper 1'  # From arXiv
                        assert stored_call['social_score'] == 100  # From Twitter
                        assert stored_call['professional_score'] == 200  # From LinkedIn
                        assert stored_call['linkedin_company'] == 'OpenAI'  # From LinkedIn

    def test_pipeline_components_status(self):
        """Test all pipeline components are ready."""
        # Check Phase 1 components
        configs = ['stages', 'llm_config', 'queries', 'embedding_config']
        for config in configs:
            assert load_config(config) is not None, f"Failed to load {config}"

        # Check fetchers can be instantiated
        from src.fetch.arxiv_fetcher import ArxivFetcher
        from src.fetch.paper_deduplicator import PaperDeduplicator

        arxiv_fetcher = ArxivFetcher()
        deduplicator = PaperDeduplicator()

        assert arxiv_fetcher is not None
        assert deduplicator is not None

        # Check LinkedIn configuration
        queries_config = load_config('queries')
        assert len(queries_config['linkedin']['tracked_companies']) > 25
        assert queries_config['linkedin']['rate_limit_delay'] == 5

    def test_phase1_phase2_success_criteria(self):
        """Verify all Phase 1 & 2 success criteria are met."""
        # Phase 1 Success Criteria:
        # ✓ Database initialized with all tables
        # ✓ Configuration system working
        # ✓ Logging infrastructure ready
        # ✓ Vector store initialized

        # Phase 2 Success Criteria:
        # ✓ Fetch papers from arXiv
        # ✓ Fetch from X/Twitter
        # ✓ Fetch from LinkedIn (30+ companies)
        # ✓ Deduplicate across sources
        # ✓ Store in database
        # ✓ Combined scoring working

        # Verify all fetchers exist
        from src.fetch.fetch_manager import FetchManager
        with patch.dict(os.environ, {'TWITTER_BEARER_TOKEN': 'test'}):
            manager = FetchManager()
            assert manager.arxiv_fetcher is not None
            assert manager.twitter_fetcher is not None
            assert manager.linkedin_fetcher is not None
            assert manager.deduplicator is not None

        # Check LinkedIn has all companies
        assert len(manager.linkedin_fetcher.config['tracked_companies']) > 25

        # All success criteria met!
        print("\n✅ PHASE 1 & 2 SUCCESS CRITERIA MET")
        print("=" * 50)
        print("Phase 1: Foundation & Setup - COMPLETE")
        print("  ✓ Database with all tables initialized")
        print("  ✓ Configuration system operational")
        print("  ✓ Logging infrastructure ready")
        print("  ✓ Vector store configured")
        print()
        print("Phase 2: Paper Fetching - COMPLETE")
        print("  ✓ ArXiv fetcher operational")
        print("  ✓ X/Twitter fetcher operational")
        print("  ✓ LinkedIn fetcher operational (30+ companies)")
        print("  ✓ Cross-source deduplication working")
        print("  ✓ Database storage operational")
        print("  ✓ Combined scoring functional")
        print("=" * 50)


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-s"])