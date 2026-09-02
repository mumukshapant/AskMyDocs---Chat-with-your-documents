"""Query-based retrieval from the vector store."""

from typing import Any, Dict, List

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
            vector_store: Vector store containing document embeddings, eg Chunk 1 -> [0.1, 0.4, ...] stored in ChromaDB
            embedding_manager: Used to convert user queries into embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager  # Used to convert user queries into embeddings. Ex : "What is a transformer model?" becomes [0.42, -0.15, 0.91, ...]

    '''
    Example :
    STEP 1 : Stored chunks are :
        Chunk A: "Machine learning is a subset of AI"
        Chunk B: "Neural networks use weighted connections"
        Chunk C: "The weather today is sunny"

    STEP 2 : Search Chroma DB
        results = self.vector_store.collection.query(..) performs vector similarity search.
        Query: "What is machine learning?"

        The vector database might return the following ranked by similarity:
        Chunk A
        Chunk B
        Chunk C

    STEP 3  : Chroma returns something like:
        {
            'documents': [[...]],
            'metadatas': [[...]],
            'distances': [[...]],
            'ids': [[...]]
        }

    STEP 4: Extract Results ( because Chroma returns a nested list )
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        After extraction: documents = [
                                        "Machine learning is...",
                                        "Neural networks are..."
                                        ]

    STEP 5 : Loop through Matches -- Processes each retrieved chunk.

    STEP 6: Convert Distance to Similarity -- Higher similarity is better.
            This is because Chroma returns distance, not similarity.

            Example:
            Distance    Similarity
            0.05           0.95
            0.10           0.90
            0.50           0.50

    STEP 7: Apply Threshold
    '''

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store --- This performs vector similarity search.
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def retrieve_per_file(self, query: str, top_k_per_file: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve top-matching chunks separately from each indexed file, then combine.
        Ensures every uploaded document is represented, regardless of its length
        relative to the others (plain top-k search favors longer documents).
        """
        all_docs = self.vector_store.collection.get()
        source_files = sorted(set(m.get("source_file") for m in all_docs["metadatas"]))

        combined = []
        for source_file in source_files:
            query_embedding = self.embedding_manager.generate_embeddings([query])[0]
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k_per_file,
                where={"source_file": source_file}
            )

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        combined.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                        })

        return combined
