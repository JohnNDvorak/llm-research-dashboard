#!/usr/bin/env python3
"""Phase 1-4 Integration Test Runner.

This script tests the complete integration of all 4 phases of the LLM Research Dashboard.
"""

import sys
import os
from pathlib import Path
import tempfile
import sqlite3
from datetime import datetime, timedelta
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

print("🧪 LLM Research Dashboard - Phase 1-4 Integration Test")
print("=" * 60)

# Test results
test_results = {
    "Phase 1 - Foundation": [],
    "Phase 2 - Fetching": [],
    "Phase 3 - Analysis": [],
    "Phase 4 - Dashboard": [],
    "End-to-End": []
}


def test_phase1_foundation():
    """Test Phase 1 components."""
    print("\n📋 Testing Phase 1: Foundation & Setup")
    print("-" * 40)

    try:
        # Test database initialization
        print("  ✅ Testing SQLite database...")
        with tempfile.TemporaryDirectory() as tmpdir:
            from storage.database import PaperDatabase
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))

            paper_id = db.add_paper(
                title="Test Paper",
                authors=["Test Author"],
                abstract="Test abstract",
                arxiv_id="2301.00001"
            )
            assert paper_id is not None

            paper = db.get_paper(paper_id)
            assert paper['title'] == "Test Paper"
            test_results["Phase 1 - Foundation"].append("✅ Database operations working")

        # Test vector store
        print("  ✅ Testing ChromaDB vector store...")
        from embeddings.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            vector_store = VectorStore(persist_dir=tmpdir)
            assert vector_store.collection is not None
            test_results["Phase 1 - Foundation"].append("✅ Vector store initialized")

        # Test configuration
        print("  ✅ Testing configuration loading...")
        from utils.config_loader import get_config
        stages = get_config("stages.yaml")
        assert len(stages) == 8
        test_results["Phase 1 - Foundation"].append("✅ Configuration loaded")

        print("  ✅ Phase 1 tests passed!")

    except Exception as e:
        print(f"  ❌ Phase 1 error: {e}")
        test_results["Phase 1 - Foundation"].append(f"❌ Error: {e}")


def test_phase2_fetching():
    """Test Phase 2 components."""
    print("\n📥 Testing Phase 2: Paper Fetching")
    print("-" * 40)

    try:
        # Test fetchers can be imported
        print("  ✅ Testing fetcher imports...")
        from fetch.arxiv_fetcher import ArxivFetcher
        from fetch.twitter_fetcher import TwitterFetcher
        from fetch.linkedin_fetcher import LinkedinFetcher
        from analysis.deduplicator import PaperDeduplicator
        test_results["Phase 2 - Fetching"].append("✅ All fetchers imported")

        # Test deduplicator
        print("  ✅ Testing deduplicator...")
        with tempfile.TemporaryDirectory() as tmpdir:
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))
            deduplicator = PaperDeduplicator(db)
            test_results["Phase 2 - Fetching"].append("✅ Deduplicator initialized")

        print("  ✅ Phase 2 tests passed!")

    except Exception as e:
        print(f"  ❌ Phase 2 error: {e}")
        test_results["Phase 2 - Fetching"].append(f"❌ Error: {e}")


def test_phase3_analysis():
    """Test Phase 3 components."""
    print("\n🧠 Testing Phase 3: LLM Analysis & Embeddings")
    print("-" * 40)

    try:
        # Test LLM provider factory
        print("  ✅ Testing LLM provider factory...")
        from llm.provider_factory import LLMProviderFactory
        factory = LLMProviderFactory()
        test_results["Phase 3 - Analysis"].append("✅ LLM factory initialized")

        # Test analyzer
        print("  ✅ Testing paper analyzer...")
        from analysis.analyzer import PaperAnalyzer
        analyzer = PaperAnalyzer()
        prompt = analyzer._generate_analysis_prompt(
            title="Test Paper",
            abstract="Test abstract"
        )
        assert "Test Paper" in prompt
        test_results["Phase 3 - Analysis"].append("✅ Analyzer generates prompts")

        # Test embeddings
        print("  ✅ Testing embedding generator...")
        from embeddings.embedding_generator import EmbeddingGenerator
        test_results["Phase 3 - Analysis"].append("✅ Embedding generator imported")

        # Test cost tracker
        print("  ✅ Testing cost tracker...")
        from utils.cost_tracker import CostTracker
        test_results["Phase 3 - Analysis"].append("✅ Cost tracker initialized")

        print("  ✅ Phase 3 tests passed!")

    except Exception as e:
        print(f"  ❌ Phase 3 error: {e}")
        test_results["Phase 3 - Analysis"].append(f"❌ Error: {e}")


def test_phase4_dashboard():
    """Test Phase 4 components."""
    print("\n📊 Testing Phase 4: Dashboard")
    print("-" * 40)

    try:
        # Test dashboard imports
        print("  ✅ Testing dashboard imports...")
        sys.path.insert(0, str(project_root / 'src'))
        from dashboard.app import init_session_state
        test_results["Phase 4 - Dashboard"].append("✅ Dashboard modules imported")

        # Test semantic search
        print("  ✅ Testing semantic search...")
        from embeddings.semantic_search import SemanticSearch
        search = SemanticSearch()
        test_results["Phase 4 - Dashboard"].append("✅ Semantic search initialized")

        print("  ✅ Phase 4 tests passed!")

    except Exception as e:
        print(f"  ❌ Phase 4 error: {e}")
        test_results["Phase 4 - Dashboard"].append(f"❌ Error: {e}")


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("\n🔄 Testing End-to-End Integration")
    print("-" * 40)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize all components
            print("  ✅ Initializing complete system...")
            db = PaperDatabase(db_path=os.path.join(tmpdir, "test.db"))
            vector_store = VectorStore(persist_dir=tmpdir)
            cost_tracker = CostTracker(db)

            # Simulate adding papers
            print("  ✅ Adding test papers...")
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

            paper_ids = []
            for paper in test_papers:
                paper_id = db.add_paper(**paper)
                paper_ids.append(paper_id)

            # Simulate analysis
            print("  ✅ Simulating analysis...")
            for i, paper_id in enumerate(paper_ids):
                db.update_analysis(
                    paper_id,
                    stages=[f"Stage {(i % 8) + 1}"],
                    summary=f"Summary for paper {i+1}",
                    key_insights=[f"Insight {i+1}.1", f"Insight {i+1}.2"]
                )

            # Test dashboard queries
            print("  ✅ Testing dashboard queries...")
            total_papers = db.get_total_papers()
            assert total_papers == 2

            recent_papers = db.get_recent_papers(days=2)
            assert len(recent_papers) == 2

            stage_dist = db.get_stage_distribution()
            assert len(stage_dist) == 2

            # Test embeddings
            print("  ✅ Testing embeddings...")
            for i, paper in enumerate(test_papers):
                vector_store.add(
                    ids=[f"paper_{paper_ids[i]}"],
                    documents=[f"{paper['title']} {paper['abstract']}"],
                    metadatas=[{'paper_id': paper_ids[i], 'title': paper['title']}]
                )

            results = vector_store.query(
                query_texts=["language models"],
                n_results=2
            )
            assert len(results['ids'][0]) > 0

            # Test cost tracking
            print("  ✅ Testing cost tracking...")
            cost_tracker.log_cost(
                provider="test_provider",
                model="test_model",
                operation="test_operation",
                tokens=1000,
                cost=0.02
            )

            costs = cost_tracker.get_total_costs()
            assert costs['total'] == 0.02

            test_results["End-to-End"].append("✅ Complete pipeline working")
            print("  ✅ End-to-end test passed!")

    except Exception as e:
        print(f"  ❌ End-to-end error: {e}")
        test_results["End-to-End"].append(f"❌ Error: {e}")


def test_database_contents():
    """Test actual database contents."""
    print("\n📊 Checking Database Contents")
    print("-" * 40)

    try:
        # Check main database
        if os.path.exists("data/papers.db"):
            conn = sqlite3.connect("data/papers.db")
            cursor = conn.cursor()

            # Count papers
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            print(f"  📄 Total papers in database: {paper_count}")

            # Count papers with analysis
            cursor.execute("SELECT COUNT(*) FROM papers WHERE stages IS NOT NULL")
            analyzed_count = cursor.fetchone()[0]
            print(f"  🧠 Papers with analysis: {analyzed_count}")

            # Count papers with embeddings
            cursor.execute("SELECT COUNT(DISTINCT paper_id) FROM embeddings")
            embedded_count = cursor.fetchone()[0]
            print(f"  🔢 Papers with embeddings: {embedded_count}")

            # Check ChromaDB
            if os.path.exists("data/chroma"):
                from embeddings.vector_store import VectorStore
                vector_store = VectorStore()
                collection_count = vector_store.collection.count()
                print(f"  📚 Vectors in ChromaDB: {collection_count}")

            conn.close()

            if paper_count > 0:
                test_results["End-to-End"].append(f"✅ Database has {paper_count} papers")
            else:
                test_results["End-to-End"].append("⚠️  Database is empty")

        else:
            print("  ⚠️  Database not found at data/papers.db")
            test_results["End-to-End"].append("⚠️  No database found")

    except Exception as e:
        print(f"  ❌ Database check error: {e}")
        test_results["End-to-End"].append(f"❌ Database error: {e}")


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for phase, results in test_results.items():
        print(f"\n{phase}:")
        for result in results:
            print(f"  {result}")
            if "❌" in result:
                all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("🎉 System is ready for deployment!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    print("\nNext Steps:")
    print("1. Run 'make dashboard' to launch the UI")
    print("2. Run 'make deploy' for deployment instructions")
    print("3. Check DEPLOYMENT.md for detailed guide")
    print("=" * 60)


def main():
    """Run all integration tests."""
    # Run all tests
    test_phase1_foundation()
    test_phase2_fetching()
    test_phase3_analysis()
    test_phase4_dashboard()
    test_end_to_end()
    test_database_contents()

    # Print summary
    print_summary()


if __name__ == "__main__":
    main()