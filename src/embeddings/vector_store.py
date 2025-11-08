"""ChromaDB vector store interface."""


class VectorStore:
    """Interface to ChromaDB for storing and querying embeddings."""

    def __init__(self, persist_directory: str = "data/chroma"):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path to ChromaDB storage
        """
        self.persist_directory = persist_directory

    def add_paper(self, paper_id: str, embedding, metadata: dict):
        """Add a paper embedding to the vector store."""
        # TODO: Implement ChromaDB insertion
        pass

    def search_similar(self, query_embedding, n_results: int = 10):
        """Search for similar papers by embedding."""
        # TODO: Implement similarity search
        pass
