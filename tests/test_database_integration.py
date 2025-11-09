"""Integration tests for SQLite and ChromaDB databases."""

import pytest
import tempfile
import shutil
from pathlib import Path
from src.storage.paper_db import PaperDB
from src.embeddings.vector_store import VectorStore


class TestSQLiteIntegration:
    """Integration tests for SQLite database."""

    def setup_method(self):
        """Create temporary database for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.db")
        self.db = PaperDB(db_path=self.db_path)

    def teardown_method(self):
        """Clean up temporary database."""
        self.db.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_tables_and_insert(self):
        """Test creating tables and inserting a paper."""
        # Create tables
        self.db.create_tables()

        # Insert a paper
        paper = {
            'id': 'arxiv:2401.00001',
            'title': 'Test Paper',
            'abstract': 'This is a test abstract',
            'authors': ['Alice', 'Bob'],
            'source': 'arxiv',
            'social_score': 100
        }

        paper_id = self.db.insert_paper(paper)
        assert paper_id == 'arxiv:2401.00001'

        # Retrieve the paper
        retrieved = self.db.get_paper(paper_id)
        assert retrieved is not None
        assert retrieved['title'] == 'Test Paper'
        assert retrieved['authors'] == ['Alice', 'Bob']
        assert retrieved['social_score'] == 100

    def test_full_crud_workflow(self):
        """Test complete CRUD workflow."""
        self.db.create_tables()

        # Create
        paper = {
            'id': 'test:001',
            'title': 'Original Title',
            'abstract': 'Original abstract',
            'stages': ['Architecture Design', 'Pre-Training'],
            'social_score': 50
        }
        self.db.insert_paper(paper)

        # Read
        retrieved = self.db.get_paper('test:001')
        assert retrieved['title'] == 'Original Title'
        assert retrieved['stages'] == ['Architecture Design', 'Pre-Training']

        # Update
        updates = {
            'title': 'Updated Title',
            'social_score': 150
        }
        result = self.db.update_paper('test:001', updates)
        assert result is True

        # Verify update
        updated = self.db.get_paper('test:001')
        assert updated['title'] == 'Updated Title'
        assert updated['social_score'] == 150
        assert updated['stages'] == ['Architecture Design', 'Pre-Training']  # Unchanged

        # Delete
        deleted = self.db.delete_paper('test:001')
        assert deleted is True

        # Verify deletion
        assert self.db.get_paper('test:001') is None

    def test_get_all_papers_with_filters(self):
        """Test retrieving papers with filtering."""
        self.db.create_tables()

        # Insert multiple papers
        papers = [
            {'id': f'test:{i}', 'title': f'Paper {i}', 'abstract': 'Abstract',
             'source': 'arxiv' if i % 2 == 0 else 'twitter', 'analyzed': i < 3}
            for i in range(10)
        ]

        for paper in papers:
            self.db.insert_paper(paper)

        # Test: Get all papers
        all_papers = self.db.get_all_papers()
        assert len(all_papers) == 10

        # Test: Filter by source
        arxiv_papers = self.db.get_all_papers(filters={'source': 'arxiv'})
        assert len(arxiv_papers) == 5

        # Test: Filter by analyzed
        analyzed_papers = self.db.get_all_papers(filters={'analyzed': True})
        assert len(analyzed_papers) == 3

        # Test: Pagination
        first_page = self.db.get_all_papers(limit=5)
        assert len(first_page) == 5

        second_page = self.db.get_all_papers(limit=5, offset=5)
        assert len(second_page) == 5

    def test_paper_exists_check(self):
        """Test checking if paper exists."""
        self.db.create_tables()

        # Paper doesn't exist yet
        assert self.db.paper_exists('test:001') is False

        # Insert paper
        paper = {'id': 'test:001', 'title': 'Test', 'abstract': 'Abstract'}
        self.db.insert_paper(paper)

        # Now it exists
        assert self.db.paper_exists('test:001') is True

    def test_paper_count(self):
        """Test counting papers."""
        self.db.create_tables()

        # Initial count
        assert self.db.get_paper_count() == 0

        # Insert papers
        for i in range(5):
            paper = {
                'id': f'test:{i}',
                'title': f'Paper {i}',
                'abstract': 'Abstract',
                'source': 'arxiv' if i < 3 else 'twitter'
            }
            self.db.insert_paper(paper)

        # Total count
        assert self.db.get_paper_count() == 5

        # Count with filter
        assert self.db.get_paper_count(filters={'source': 'arxiv'}) == 3
        assert self.db.get_paper_count(filters={'source': 'twitter'}) == 2

    def test_cost_tracking(self):
        """Test inserting cost records."""
        self.db.create_tables()

        # Insert cost record
        cost_data = {
            'provider': 'xai',
            'model': 'grok-4-fast-reasoning',
            'paper_id': 'test:001',
            'operation_type': 'analysis',
            'input_tokens': 1000,
            'output_tokens': 500,
            'cost': 0.0007
        }

        record_id = self.db.insert_cost_record(cost_data)
        assert record_id > 0

    def test_context_manager(self):
        """Test using PaperDB as context manager."""
        # Use context manager
        with PaperDB(db_path=self.db_path) as db:
            db.create_tables()
            paper = {'id': 'test:001', 'title': 'Test', 'abstract': 'Abstract'}
            db.insert_paper(paper)

        # Connection should be closed, but we can reconnect
        with PaperDB(db_path=self.db_path) as db:
            retrieved = db.get_paper('test:001')
            assert retrieved is not None


class TestChromaDBIntegration:
    """Integration tests for ChromaDB vector store."""

    def setup_method(self):
        """Create temporary vector store for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = VectorStore(persist_directory=self.temp_dir)

    def teardown_method(self):
        """Clean up temporary vector store."""
        try:
            self.store.disconnect()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_and_retrieve_paper(self):
        """Test adding and retrieving a paper by ID."""
        self.store.connect()

        # Add paper
        embedding = [0.1] * 1536
        metadata = {
            'title': 'Test Paper',
            'stages': ['Architecture Design'],
            'source': 'arxiv'
        }
        document = "Test Paper [SEP] This is a test abstract"

        self.store.add_paper('test:001', embedding, metadata, document)

        # Retrieve by ID
        result = self.store.get_by_id('test:001')
        assert result is not None
        assert result['id'] == 'test:001'
        assert len(result['embedding']) == 1536
        assert 'Test Paper' in result['metadata']['title']

    def test_similarity_search(self):
        """Test semantic similarity search."""
        self.store.connect()

        # Add multiple papers with different embeddings
        papers = [
            ('paper:1', [0.9] + [0.1] * 1535, {'title': 'Paper 1', 'topic': 'attention'}),
            ('paper:2', [0.8] + [0.2] * 1535, {'title': 'Paper 2', 'topic': 'training'}),
            ('paper:3', [0.7] + [0.3] * 1535, {'title': 'Paper 3', 'topic': 'evaluation'}),
        ]

        for paper_id, embedding, metadata in papers:
            self.store.add_paper(paper_id, embedding, metadata, f"Document for {paper_id}")

        # Search with query similar to paper:1
        query_embedding = [0.85] + [0.15] * 1535
        results = self.store.search_similar(query_embedding, n_results=2)

        assert len(results['ids']) == 2
        # Should find paper:1 as most similar
        assert 'paper:1' in results['ids']

    def test_batch_add_papers(self):
        """Test adding papers in batch."""
        self.store.connect()

        # Prepare batch data
        paper_ids = [f'batch:{ i}' for i in range(10)]
        embeddings = [[0.1 * i] * 1536 for i in range(10)]
        metadatas = [{'title': f'Paper {i}', 'index': i} for i in range(10)]
        documents = [f'Document {i}' for i in range(10)]

        # Add batch
        self.store.add_papers_batch(paper_ids, embeddings, metadatas, documents)

        # Verify count
        assert self.store.count() == 10

        # Verify individual papers exist
        assert self.store.paper_exists('batch:0')
        assert self.store.paper_exists('batch:9')

    def test_update_paper(self):
        """Test updating paper in vector store."""
        self.store.connect()

        # Add initial paper
        embedding = [0.5] * 1536
        metadata = {'title': 'Original', 'version': 1}
        self.store.add_paper('test:001', embedding, metadata, 'Original doc')

        # Update metadata only
        new_metadata = {'title': 'Updated', 'version': 2}
        self.store.update_paper('test:001', metadata=new_metadata)

        # Verify update
        result = self.store.get_by_id('test:001')
        assert 'Updated' in result['metadata']['title']

    def test_delete_paper(self):
        """Test deleting paper from vector store."""
        self.store.connect()

        # Add paper
        embedding = [0.1] * 1536
        metadata = {'title': 'Test'}
        self.store.add_paper('test:001', embedding, metadata)

        # Verify it exists
        assert self.store.paper_exists('test:001')

        # Delete it
        self.store.delete_paper('test:001')

        # Verify deletion
        assert not self.store.paper_exists('test:001')

    def test_filtered_similarity_search(self):
        """Test similarity search with metadata filters."""
        self.store.connect()

        # Add papers with different sources
        papers = [
            ('arxiv:1', [0.1] * 1536, {'title': 'ArXiv Paper 1', 'source': 'arxiv'}),
            ('arxiv:2', [0.2] * 1536, {'title': 'ArXiv Paper 2', 'source': 'arxiv'}),
            ('twitter:1', [0.15] * 1536, {'title': 'Twitter Paper', 'source': 'twitter'}),
        ]

        for paper_id, embedding, metadata in papers:
            self.store.add_paper(paper_id, embedding, metadata)

        # Search with filter for arxiv only
        query_embedding = [0.15] * 1536
        results = self.store.search_similar(
            query_embedding,
            n_results=10,
            where={'source': 'arxiv'}
        )

        # Should only find arxiv papers
        assert len(results['ids']) == 2
        assert all('arxiv' in pid for pid in results['ids'])

    def test_context_manager(self):
        """Test using VectorStore as context manager."""
        # Use context manager
        with VectorStore(persist_directory=self.temp_dir) as store:
            embedding = [0.1] * 1536
            metadata = {'title': 'Test'}
            store.add_paper('test:001', embedding, metadata)
            assert store.count() == 1

        # Should be able to reconnect and access data
        with VectorStore(persist_directory=self.temp_dir) as store:
            assert store.count() == 1
            assert store.paper_exists('test:001')


class TestDatabasesIntegration:
    """Integration tests for SQLite and ChromaDB working together."""

    def setup_method(self):
        """Create temporary databases."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = str(Path(self.temp_dir) / "test.db")
        self.chroma_path = str(Path(self.temp_dir) / "chroma")

        self.paper_db = PaperDB(db_path=self.db_path)
        self.vector_store = VectorStore(persist_directory=self.chroma_path)

    def teardown_method(self):
        """Clean up."""
        self.paper_db.close()
        try:
            self.vector_store.disconnect()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_paper_workflow_both_databases(self):
        """Test complete paper workflow across both databases."""
        # Initialize databases
        self.paper_db.create_tables()
        self.vector_store.connect()

        # Paper data
        paper_id = 'arxiv:2401.00001'
        paper = {
            'id': paper_id,
            'title': 'Attention Is All You Need',
            'abstract': 'We propose a new architecture based on attention mechanisms',
            'authors': ['Vaswani', 'Shazeer'],
            'source': 'arxiv',
            'stages': ['Architecture Design'],
            'social_score': 1000
        }

        # Insert into SQLite
        self.paper_db.insert_paper(paper)

        # Generate and store embedding
        embedding = [0.5] * 1536  # Mock embedding
        metadata = {
            'title': paper['title'],
            'stages': str(paper['stages']),
            'source': paper['source']
        }
        document = f"{paper['title']} [SEP] {paper['abstract']}"

        self.vector_store.add_paper(paper_id, embedding, metadata, document)

        # Verify in both databases
        sql_paper = self.paper_db.get_paper(paper_id)
        assert sql_paper is not None
        assert sql_paper['title'] == paper['title']

        vector_paper = self.vector_store.get_by_id(paper_id)
        assert vector_paper is not None
        assert len(vector_paper['embedding']) == 1536

        # Update in SQLite
        self.paper_db.update_paper(paper_id, {
            'embedding_generated': True,
            'chroma_id': paper_id
        })

        # Verify update
        updated_paper = self.paper_db.get_paper(paper_id)
        assert updated_paper['embedding_generated'] == 1  # SQLite stores as int
        assert updated_paper['chroma_id'] == paper_id

    def test_search_and_retrieve_workflow(self):
        """Test searching in ChromaDB and retrieving from SQLite."""
        self.paper_db.create_tables()
        self.vector_store.connect()

        # Add 5 papers to both databases
        for i in range(5):
            paper_id = f'test:{i}'
            paper = {
                'id': paper_id,
                'title': f'Paper {i}',
                'abstract': f'Abstract for paper {i}',
                'source': 'arxiv',
                'social_score': i * 10
            }

            self.paper_db.insert_paper(paper)

            embedding = [0.1 * i] * 1536
            metadata = {'title': paper['title'], 'index': i}
            self.vector_store.add_paper(paper_id, embedding, metadata)

        # Search in ChromaDB
        query_embedding = [0.2] * 1536
        results = self.vector_store.search_similar(query_embedding, n_results=3)

        assert len(results['ids']) == 3

        # Retrieve full details from SQLite
        for paper_id in results['ids']:
            paper = self.paper_db.get_paper(paper_id)
            assert paper is not None
            assert 'title' in paper
            assert 'social_score' in paper
