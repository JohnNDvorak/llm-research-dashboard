"""Unit tests for embedding modules."""

import pytest
from src.embeddings.vector_store import VectorStore
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.semantic_search import SemanticSearch


class TestVectorStore:
    """Test suite for VectorStore class."""

    def test_vector_store_can_be_instantiated(self):
        """Test that VectorStore can be instantiated."""
        store = VectorStore()
        assert store is not None

    def test_vector_store_with_default_path(self):
        """Test VectorStore initialization with default path."""
        store = VectorStore()
        assert store.persist_directory == "data/chroma"

    def test_vector_store_with_custom_path(self):
        """Test VectorStore initialization with custom path."""
        custom_path = "test/vectors"
        store = VectorStore(persist_directory=custom_path)
        assert store.persist_directory == custom_path

    def test_vector_store_stores_directory(self):
        """Test that VectorStore stores persist directory."""
        store = VectorStore()
        assert hasattr(store, 'persist_directory')
        assert isinstance(store.persist_directory, str)

    def test_vector_store_has_add_paper_method(self):
        """Test that VectorStore has add_paper method."""
        store = VectorStore()
        assert hasattr(store, 'add_paper')
        assert callable(store.add_paper)

    def test_vector_store_has_search_similar_method(self):
        """Test that VectorStore has search_similar method."""
        store = VectorStore()
        assert hasattr(store, 'search_similar')
        assert callable(store.search_similar)

    def test_add_paper_signature(self):
        """Test that add_paper has correct signature."""
        import inspect
        sig = inspect.signature(VectorStore.add_paper)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'paper_id' in params
        assert 'embedding' in params
        assert 'metadata' in params

    def test_search_similar_signature(self):
        """Test that search_similar has correct signature."""
        import inspect
        sig = inspect.signature(VectorStore.search_similar)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'query_embedding' in params
        assert 'n_results' in params


class TestVectorStoreOperations:
    """Test suite for VectorStore operations."""

    def test_add_paper_with_embedding(self):
        """Test add_paper with embedding vector."""
        store = VectorStore()
        embedding = [0.1] * 1536  # 1536-dimensional vector
        metadata = {"title": "Test Paper"}
        try:
            store.add_paper("arxiv:2401.00001", embedding, metadata)
        except TypeError as e:
            pytest.fail(f"add_paper should accept these parameters: {e}")

    def test_add_paper_with_empty_metadata(self):
        """Test add_paper with empty metadata."""
        store = VectorStore()
        embedding = [0.1] * 1536
        store.add_paper("arxiv:2401.00001", embedding, {})

    def test_add_paper_with_rich_metadata(self):
        """Test add_paper with rich metadata."""
        store = VectorStore()
        embedding = [0.1] * 1536
        metadata = {
            "title": "Paper Title",
            "abstract": "Abstract text",
            "authors": ["Author 1"],
            "stage": "Architecture Design",
            "score": 95.5
        }
        store.add_paper("arxiv:2401.00001", embedding, metadata)

    def test_search_similar_with_embedding(self):
        """Test search_similar with query embedding."""
        store = VectorStore()
        query_embedding = [0.2] * 1536
        try:
            store.search_similar(query_embedding)
        except TypeError as e:
            pytest.fail(f"search_similar should accept embedding: {e}")

    def test_search_similar_with_n_results(self):
        """Test search_similar with custom n_results."""
        store = VectorStore()
        query_embedding = [0.2] * 1536
        store.search_similar(query_embedding, n_results=5)

    def test_search_similar_default_n_results(self):
        """Test search_similar with default n_results."""
        store = VectorStore()
        query_embedding = [0.2] * 1536
        store.search_similar(query_embedding)


class TestVectorStorePaths:
    """Test suite for various VectorStore path scenarios."""

    def test_vector_store_with_relative_path(self):
        """Test VectorStore with relative path."""
        store = VectorStore(persist_directory="data/chroma")
        assert store.persist_directory == "data/chroma"

    def test_vector_store_with_absolute_path(self):
        """Test VectorStore with absolute path."""
        store = VectorStore(persist_directory="/tmp/chroma")
        assert store.persist_directory == "/tmp/chroma"

    def test_vector_store_with_nested_path(self):
        """Test VectorStore with nested directory path."""
        store = VectorStore(persist_directory="data/vectors/chroma/v1")
        assert store.persist_directory == "data/vectors/chroma/v1"


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class."""

    def test_embedding_generator_can_be_instantiated(self):
        """Test that EmbeddingGenerator can be instantiated."""
        gen = EmbeddingGenerator()
        assert gen is not None

    def test_embedding_generator_with_default_provider(self):
        """Test EmbeddingGenerator with default provider."""
        gen = EmbeddingGenerator()
        assert gen.provider == "openai"

    def test_embedding_generator_with_openai_provider(self):
        """Test EmbeddingGenerator with openai provider."""
        gen = EmbeddingGenerator(provider="openai")
        assert gen.provider == "openai"

    def test_embedding_generator_with_local_provider(self):
        """Test EmbeddingGenerator with local provider."""
        gen = EmbeddingGenerator(provider="local")
        assert gen.provider == "local"

    def test_embedding_generator_stores_provider(self):
        """Test that EmbeddingGenerator stores provider."""
        gen = EmbeddingGenerator()
        assert hasattr(gen, 'provider')
        assert isinstance(gen.provider, str)

    def test_embedding_generator_has_generate_method(self):
        """Test that EmbeddingGenerator has generate method."""
        gen = EmbeddingGenerator()
        assert hasattr(gen, 'generate')
        assert callable(gen.generate)

    def test_generate_signature(self):
        """Test that generate has correct signature."""
        import inspect
        sig = inspect.signature(EmbeddingGenerator.generate)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'text' in params

    def test_generate_return_type(self):
        """Test that generate has correct return type annotation."""
        import inspect
        sig = inspect.signature(EmbeddingGenerator.generate)
        # Should return List[float]
        assert sig.return_annotation != inspect.Signature.empty


class TestEmbeddingGeneration:
    """Test suite for embedding generation scenarios."""

    def test_generate_with_simple_text(self):
        """Test generate with simple text."""
        gen = EmbeddingGenerator()
        try:
            gen.generate("simple text")
        except TypeError as e:
            pytest.fail(f"generate should accept text: {e}")

    def test_generate_with_long_text(self):
        """Test generate with long text."""
        gen = EmbeddingGenerator()
        text = "This is a long paper abstract. " * 100
        gen.generate(text)

    def test_generate_with_empty_text(self):
        """Test generate with empty text."""
        gen = EmbeddingGenerator()
        gen.generate("")

    def test_generate_with_unicode(self):
        """Test generate with unicode text."""
        gen = EmbeddingGenerator()
        gen.generate("LLM Training 中文 🤖")

    def test_generate_with_special_characters(self):
        """Test generate with special characters."""
        gen = EmbeddingGenerator()
        gen.generate("Text with $pecial ch@rs! and 'quotes'")

    def test_generate_with_paper_title(self):
        """Test generate with paper title."""
        gen = EmbeddingGenerator()
        gen.generate("Attention Is All You Need")

    def test_generate_with_paper_abstract(self):
        """Test generate with paper abstract."""
        gen = EmbeddingGenerator()
        abstract = "We propose a new transformer architecture..."
        gen.generate(abstract)

    def test_generate_with_combined_text(self):
        """Test generate with combined title + abstract."""
        gen = EmbeddingGenerator()
        text = "Attention Is All You Need [SEP] We propose a new architecture"
        gen.generate(text)


class TestEmbeddingProviders:
    """Test suite for different embedding providers."""

    def test_openai_provider_initialization(self):
        """Test OpenAI provider initialization."""
        gen = EmbeddingGenerator(provider="openai")
        assert gen.provider == "openai"

    def test_local_provider_initialization(self):
        """Test local provider initialization."""
        gen = EmbeddingGenerator(provider="local")
        assert gen.provider == "local"

    def test_custom_provider_initialization(self):
        """Test custom provider initialization."""
        gen = EmbeddingGenerator(provider="voyage")
        assert gen.provider == "voyage"

    def test_generate_with_openai(self):
        """Test generate with OpenAI provider."""
        gen = EmbeddingGenerator(provider="openai")
        gen.generate("test text")

    def test_generate_with_local(self):
        """Test generate with local provider."""
        gen = EmbeddingGenerator(provider="local")
        gen.generate("test text")


class TestSemanticSearch:
    """Test suite for SemanticSearch class."""

    def test_semantic_search_can_be_instantiated(self):
        """Test that SemanticSearch can be instantiated."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        assert search is not None

    def test_semantic_search_stores_vector_store(self):
        """Test that SemanticSearch stores vector store."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        assert hasattr(search, 'vector_store')
        assert search.vector_store == vector_store

    def test_semantic_search_stores_embedding_generator(self):
        """Test that SemanticSearch stores embedding generator."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        assert hasattr(search, 'embedding_generator')
        assert search.embedding_generator == embedding_gen

    def test_semantic_search_has_search_method(self):
        """Test that SemanticSearch has search method."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        assert hasattr(search, 'search')
        assert callable(search.search)

    def test_search_signature(self):
        """Test that search has correct signature."""
        import inspect
        sig = inspect.signature(SemanticSearch.search)
        params = list(sig.parameters.keys())
        assert 'self' in params
        assert 'query' in params
        assert 'n_results' in params


class TestSemanticSearchOperations:
    """Test suite for semantic search operations."""

    def test_search_with_simple_query(self):
        """Test search with simple query."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        try:
            search.search("attention mechanism")
        except TypeError as e:
            pytest.fail(f"search should accept query: {e}")

    def test_search_with_n_results(self):
        """Test search with custom n_results."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("LLM training", n_results=5)

    def test_search_with_default_n_results(self):
        """Test search with default n_results."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("transformer architecture")

    def test_search_with_long_query(self):
        """Test search with long query."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        query = "I'm looking for papers about direct preference optimization in large language models"
        search.search(query)

    def test_search_with_short_query(self):
        """Test search with very short query."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("DPO")

    def test_search_with_empty_query(self):
        """Test search with empty query."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("")

    def test_search_with_unicode_query(self):
        """Test search with unicode query."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("LLM 训练 🤖")


class TestSemanticSearchIntegration:
    """Integration tests for semantic search scenarios."""

    def test_search_with_openai_embeddings(self):
        """Test semantic search with OpenAI embeddings."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator(provider="openai")
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("attention mechanism", n_results=10)

    def test_search_with_local_embeddings(self):
        """Test semantic search with local embeddings."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator(provider="local")
        search = SemanticSearch(vector_store, embedding_gen)
        search.search("transformer architecture", n_results=5)

    def test_search_various_topics(self):
        """Test search with various research topics."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)

        queries = [
            "attention mechanism",
            "DPO and RLHF",
            "model quantization",
            "distributed training",
            "mixture of experts"
        ]

        for query in queries:
            search.search(query, n_results=10)

    def test_search_with_different_result_counts(self):
        """Test search with different result counts."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)

        for n in [1, 5, 10, 20, 50]:
            search.search("LLM training", n_results=n)

    def test_multiple_searches_same_instance(self):
        """Test multiple searches with same search instance."""
        vector_store = VectorStore()
        embedding_gen = EmbeddingGenerator()
        search = SemanticSearch(vector_store, embedding_gen)

        search.search("attention", n_results=5)
        search.search("training", n_results=10)
        search.search("evaluation", n_results=15)

    def test_semantic_search_realistic_workflow(self):
        """Test realistic semantic search workflow."""
        # Initialize components
        vector_store = VectorStore(persist_directory="data/chroma")
        embedding_gen = EmbeddingGenerator(provider="openai")
        search = SemanticSearch(vector_store, embedding_gen)

        # Perform searches
        results_1 = search.search("post-training methods like DPO", n_results=10)
        results_2 = search.search("transformer architecture design", n_results=5)
        results_3 = search.search("model deployment and inference", n_results=20)
