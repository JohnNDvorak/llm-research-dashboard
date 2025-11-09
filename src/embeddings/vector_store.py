"""ChromaDB vector store interface."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.utils.logger import logger


class VectorStore:
    """Interface to ChromaDB for storing and querying embeddings."""

    def __init__(self, persist_directory: str = "data/chroma"):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Path to ChromaDB storage
        """
        self.persist_directory = persist_directory
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        self.collection_name = "llm_papers"

    def connect(self) -> None:
        """Initialize ChromaDB client and collection."""
        # Create persist directory if it doesn't exist
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LLM research papers embeddings"}
        )

        logger.info(f"Connected to ChromaDB at {self.persist_directory}")
        logger.info(f"Collection '{self.collection_name}' has {self.collection.count()} documents")

    def disconnect(self) -> None:
        """Disconnect from ChromaDB (cleanup)."""
        self.collection = None
        self.client = None
        logger.info("Disconnected from ChromaDB")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def add_paper(
        self,
        paper_id: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        document: Optional[str] = None
    ) -> None:
        """
        Add a paper embedding to the vector store.

        Args:
            paper_id: Unique paper identifier
            embedding: 1536-dimensional embedding vector
            metadata: Paper metadata (title, stages, scores, etc.)
            document: Optional full text for context

        Raises:
            ValueError: If embedding dimension is incorrect
            Exception: If ChromaDB operation fails
        """
        if not self.collection:
            self.connect()

        # Validate embedding dimensions
        if not isinstance(embedding, list):
            raise ValueError("Embedding must be a list of floats")

        if len(embedding) == 0:
            raise ValueError("Embedding cannot be empty")

        # Prepare metadata (ChromaDB requires all values to be simple types)
        clean_metadata = {}
        for key, value in metadata.items():
            # Convert lists to strings
            if isinstance(value, list):
                clean_metadata[key] = str(value)
            # Convert None to empty string
            elif value is None:
                clean_metadata[key] = ""
            # Keep simple types
            elif isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)

        # Prepare document text
        doc_text = document if document else f"Paper ID: {paper_id}"

        try:
            self.collection.add(
                ids=[paper_id],
                embeddings=[embedding],
                metadatas=[clean_metadata],
                documents=[doc_text]
            )
            logger.info(f"Added paper to vector store: {paper_id}")
        except Exception as e:
            logger.error(f"Failed to add paper {paper_id} to vector store: {e}")
            raise

    def add_papers_batch(
        self,
        paper_ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: Optional[List[str]] = None
    ) -> None:
        """
        Add multiple papers in batch (more efficient).

        Args:
            paper_ids: List of paper IDs
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: Optional list of full text documents

        Raises:
            ValueError: If list lengths don't match
            Exception: If ChromaDB operation fails
        """
        if not self.collection:
            self.connect()

        # Validate inputs
        if not (len(paper_ids) == len(embeddings) == len(metadatas)):
            raise ValueError("All input lists must have the same length")

        if documents and len(documents) != len(paper_ids):
            raise ValueError("Documents list must match length of paper_ids")

        # Prepare metadata
        clean_metadatas = []
        for metadata in metadatas:
            clean_meta = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    clean_meta[key] = str(value)
                elif value is None:
                    clean_meta[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                else:
                    clean_meta[key] = str(value)
            clean_metadatas.append(clean_meta)

        # Prepare documents
        doc_texts = documents if documents else [f"Paper ID: {pid}" for pid in paper_ids]

        try:
            self.collection.add(
                ids=paper_ids,
                embeddings=embeddings,
                metadatas=clean_metadatas,
                documents=doc_texts
            )
            logger.info(f"Added {len(paper_ids)} papers to vector store in batch")
        except Exception as e:
            logger.error(f"Failed to add papers batch to vector store: {e}")
            raise

    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar papers by embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filters (e.g., {"source": "arxiv"})

        Returns:
            Dictionary with ids, distances, metadatas, and documents

        Raises:
            ValueError: If query embedding is invalid
            Exception: If ChromaDB query fails
        """
        if not self.collection:
            self.connect()

        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            raise ValueError("Query embedding must be a non-empty list")

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )

            logger.info(f"Found {len(results['ids'][0])} similar papers")

            return {
                'ids': results['ids'][0] if results['ids'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'documents': results['documents'][0] if results['documents'] else []
            }
        except Exception as e:
            logger.error(f"Failed to search similar papers: {e}")
            raise

    def get_by_id(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a paper by ID from vector store.

        Args:
            paper_id: Paper ID to retrieve

        Returns:
            Dictionary with embedding, metadata, and document, or None if not found
        """
        if not self.collection:
            self.connect()

        try:
            result = self.collection.get(
                ids=[paper_id],
                include=["embeddings", "metadatas", "documents"]
            )

            if not result['ids']:
                return None

            return {
                'id': result['ids'][0],
                'embedding': result['embeddings'][0] if result['embeddings'] is not None else None,
                'metadata': result['metadatas'][0] if result['metadatas'] is not None else {},
                'document': result['documents'][0] if result['documents'] is not None else ""
            }
        except Exception as e:
            logger.error(f"Failed to get paper {paper_id} from vector store: {e}")
            raise

    def update_paper(
        self,
        paper_id: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        document: Optional[str] = None
    ) -> None:
        """
        Update a paper in the vector store.

        Args:
            paper_id: Paper ID to update
            embedding: New embedding (optional)
            metadata: New metadata (optional)
            document: New document text (optional)

        Raises:
            Exception: If ChromaDB operation fails
        """
        if not self.collection:
            self.connect()

        # Prepare metadata
        clean_metadata = None
        if metadata:
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, list):
                    clean_metadata[key] = str(value)
                elif value is None:
                    clean_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                else:
                    clean_metadata[key] = str(value)

        try:
            self.collection.update(
                ids=[paper_id],
                embeddings=[embedding] if embedding else None,
                metadatas=[clean_metadata] if clean_metadata else None,
                documents=[document] if document else None
            )
            logger.info(f"Updated paper in vector store: {paper_id}")
        except Exception as e:
            logger.error(f"Failed to update paper {paper_id} in vector store: {e}")
            raise

    def delete_paper(self, paper_id: str) -> None:
        """
        Delete a paper from the vector store.

        Args:
            paper_id: Paper ID to delete

        Raises:
            Exception: If ChromaDB operation fails
        """
        if not self.collection:
            self.connect()

        try:
            self.collection.delete(ids=[paper_id])
            logger.info(f"Deleted paper from vector store: {paper_id}")
        except Exception as e:
            logger.error(f"Failed to delete paper {paper_id} from vector store: {e}")
            raise

    def paper_exists(self, paper_id: str) -> bool:
        """
        Check if paper exists in vector store.

        Args:
            paper_id: Paper ID to check

        Returns:
            True if paper exists, False otherwise
        """
        if not self.collection:
            self.connect()

        try:
            result = self.collection.get(ids=[paper_id])
            return len(result['ids']) > 0
        except Exception as e:
            logger.error(f"Failed to check paper existence {paper_id}: {e}")
            raise

    def count(self) -> int:
        """
        Get total count of papers in vector store.

        Returns:
            Number of papers
        """
        if not self.collection:
            self.connect()

        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count papers in vector store: {e}")
            raise

    def reset(self) -> None:
        """
        Reset (delete all data) from the collection.

        WARNING: This is destructive and cannot be undone.

        Raises:
            Exception: If ChromaDB operation fails
        """
        if not self.collection:
            self.connect()

        try:
            if self.client:
                self.client.delete_collection(name=self.collection_name)
                # Recreate empty collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "LLM research papers embeddings"}
                )
                logger.warning("Vector store collection has been reset (all data deleted)")
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            raise
