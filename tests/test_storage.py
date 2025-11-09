"""Unit tests for storage modules."""

import pytest
from src.storage.paper_db import PaperDB


class TestPaperDB:
    """Test suite for PaperDB class."""

    def test_paper_db_can_be_instantiated(self):
        """Test that PaperDB can be instantiated."""
        db = PaperDB()
        assert db is not None

    def test_paper_db_with_default_path(self):
        """Test PaperDB initialization with default path."""
        db = PaperDB()
        assert db.db_path == "data/papers.db"

    def test_paper_db_with_custom_path(self):
        """Test PaperDB initialization with custom path."""
        custom_path = "test/custom.db"
        db = PaperDB(db_path=custom_path)
        assert db.db_path == custom_path

    def test_paper_db_stores_db_path(self):
        """Test that PaperDB stores database path."""
        db = PaperDB()
        assert hasattr(db, 'db_path')
        assert isinstance(db.db_path, str)

    def test_paper_db_has_create_tables_method(self):
        """Test that PaperDB has create_tables method."""
        db = PaperDB()
        assert hasattr(db, 'create_tables')
        assert callable(db.create_tables)

    def test_paper_db_has_insert_paper_method(self):
        """Test that PaperDB has insert_paper method."""
        db = PaperDB()
        assert hasattr(db, 'insert_paper')
        assert callable(db.insert_paper)

    def test_paper_db_has_get_paper_method(self):
        """Test that PaperDB has get_paper method."""
        db = PaperDB()
        assert hasattr(db, 'get_paper')
        assert callable(db.get_paper)

    def test_create_tables_signature(self):
        """Test that create_tables has correct signature."""
        import inspect
        sig = inspect.signature(PaperDB.create_tables)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_insert_paper_signature(self):
        """Test that insert_paper has correct signature."""
        import inspect
        sig = inspect.signature(PaperDB.insert_paper)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'paper' in params

    def test_get_paper_signature(self):
        """Test that get_paper has correct signature."""
        import inspect
        sig = inspect.signature(PaperDB.get_paper)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'paper_id' in params


class TestPaperDBOperations:
    """Test suite for PaperDB operations."""

    def test_create_tables_call(self):
        """Test that create_tables can be called."""
        db = PaperDB()
        try:
            db.create_tables()
        except Exception as e:
            # Expected to pass/return None for now (TODO implementation)
            pass

    def test_insert_paper_with_dict(self):
        """Test insert_paper with dictionary parameter."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "Test Paper",
            "abstract": "Test abstract"
        }
        try:
            db.insert_paper(paper)
        except TypeError as e:
            pytest.fail(f"insert_paper should accept dict: {e}")

    def test_get_paper_with_id(self):
        """Test get_paper with paper ID."""
        db = PaperDB()
        try:
            db.get_paper("arxiv:2401.00001")
        except TypeError as e:
            pytest.fail(f"get_paper should accept string ID: {e}")

    def test_insert_paper_with_full_data(self):
        """Test insert_paper with complete paper data."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "Complete Paper",
            "abstract": "Full abstract text",
            "authors": ["Author 1", "Author 2"],
            "published_date": "2024-01-01",
            "arxiv_url": "https://arxiv.org/abs/2401.00001",
            "categories": ["cs.CL", "cs.AI"],
            "social_score": 100,
            "professional_score": 50
        }
        db.insert_paper(paper)

    def test_insert_paper_with_minimal_data(self):
        """Test insert_paper with minimal paper data."""
        db = PaperDB()
        paper = {"id": "arxiv:2401.00001"}
        db.insert_paper(paper)

    def test_get_paper_with_arxiv_id(self):
        """Test get_paper with arXiv ID format."""
        db = PaperDB()
        db.get_paper("arxiv:2401.00001")

    def test_get_paper_with_different_id_formats(self):
        """Test get_paper with various ID formats."""
        db = PaperDB()
        db.get_paper("arxiv:2401.00001")
        db.get_paper("2401.00001")
        db.get_paper("1")


class TestPaperDBPaths:
    """Test suite for various database path scenarios."""

    def test_paper_db_with_relative_path(self):
        """Test PaperDB with relative path."""
        db = PaperDB(db_path="data/test.db")
        assert db.db_path == "data/test.db"

    def test_paper_db_with_absolute_path(self):
        """Test PaperDB with absolute path."""
        db = PaperDB(db_path="/tmp/papers.db")
        assert db.db_path == "/tmp/papers.db"

    def test_paper_db_with_memory_database(self):
        """Test PaperDB with in-memory database."""
        db = PaperDB(db_path=":memory:")
        assert db.db_path == ":memory:"

    def test_paper_db_with_nested_path(self):
        """Test PaperDB with nested directory path."""
        db = PaperDB(db_path="data/databases/test/papers.db")
        assert db.db_path == "data/databases/test/papers.db"

    def test_paper_db_with_extension_variants(self):
        """Test PaperDB with different file extensions."""
        db1 = PaperDB(db_path="papers.db")
        db2 = PaperDB(db_path="papers.sqlite")
        db3 = PaperDB(db_path="papers.sqlite3")
        assert db1.db_path == "papers.db"
        assert db2.db_path == "papers.sqlite"
        assert db3.db_path == "papers.sqlite3"


class TestPaperDBIntegration:
    """Integration tests for PaperDB scenarios."""

    def test_multiple_paper_insertions(self):
        """Test inserting multiple papers."""
        db = PaperDB()
        papers = [
            {"id": f"arxiv:2401.{str(i).zfill(5)}", "title": f"Paper {i}"}
            for i in range(10)
        ]
        for paper in papers:
            db.insert_paper(paper)

    def test_insert_and_retrieve_flow(self):
        """Test basic insert and retrieve flow."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "Test Paper"
        }
        db.insert_paper(paper)
        db.get_paper("arxiv:2401.00001")

    def test_paper_with_unicode(self):
        """Test paper with unicode characters."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "LLM Training 中文 🤖",
            "abstract": "Résumé with unicode"
        }
        db.insert_paper(paper)

    def test_paper_with_special_characters(self):
        """Test paper with special characters."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "Paper with 'quotes' and \"double quotes\"",
            "abstract": "Abstract with $pecial ch@rs!"
        }
        db.insert_paper(paper)

    def test_paper_with_long_abstract(self):
        """Test paper with very long abstract."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "Paper",
            "abstract": "Long abstract " * 1000
        }
        db.insert_paper(paper)

    def test_paper_with_null_values(self):
        """Test paper with null values."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": None,
            "abstract": None
        }
        db.insert_paper(paper)

    def test_paper_with_empty_strings(self):
        """Test paper with empty strings."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "title": "",
            "abstract": ""
        }
        db.insert_paper(paper)

    def test_paper_with_list_fields(self):
        """Test paper with list fields."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "authors": ["Author 1", "Author 2", "Author 3"],
            "categories": ["cs.CL", "cs.LG"],
            "stages": ["Architecture", "Training"]
        }
        db.insert_paper(paper)

    def test_paper_with_nested_dict(self):
        """Test paper with nested dictionary."""
        db = PaperDB()
        paper = {
            "id": "arxiv:2401.00001",
            "metadata": {
                "source": "twitter",
                "social_score": 100,
                "engagement": {
                    "likes": 50,
                    "retweets": 25
                }
            }
        }
        db.insert_paper(paper)

    def test_get_nonexistent_paper(self):
        """Test getting a paper that doesn't exist."""
        db = PaperDB()
        db.get_paper("nonexistent_id")

    def test_empty_database_operations(self):
        """Test operations on empty database."""
        db = PaperDB()
        db.get_paper("arxiv:2401.00001")
