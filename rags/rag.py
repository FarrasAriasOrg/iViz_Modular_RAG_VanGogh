from data_processing import DataProcessor
from embedder import Embedder
from embedder_openai import EmbedderOpenAI

import faiss  # Import Faiss if necessary
api_key =  ""
class RAG:
    """Represents a Retrieval-Augmented Generation (RAG) model."""

    def __init__(self, name, document_path=None, db=None, num_results=5, 
                 chunk_length=256, separator_value=None, metadata=None, similarity_threshold=999):
        """Initializes the RAG model.

        Args:
            name: The name of the RAG model.
            document_path: Path to a document (if using a document source).
            db: An existing database (if using a database source).
            num_results: The maximum number of results to return.
            chunk_length: The length of text chunks when splitting a document.
            separator_value: Separator for splitting a document.
            metadata: Additional metadata about the RAG model.
            similarity_threshold: Maximum similarity distance for filtering results. 
        """
        self._chunks = None
        self.name = name
        self.similarity_threshold = similarity_threshold 
        self.encoder = None

        # Use order from metadata, with a default of 9999
        self.order = metadata.get("order", 9999) if metadata else 9999

        # Use context from metadata, with a default
        self.context = metadata.get("context", "Use this information to answer: ") if metadata else "Use this information to answer: "

        if document_path:
            processor = DataProcessor()  # Assuming you have a DataProcessor class
            if separator_value:
                self._chunks = processor.split_text(f"{document_path}.txt", separator=separator_value)
            else:
                self._chunks = processor.split_text(f"{document_path}.txt", chunk_length=chunk_length)
        elif db is not None:
            self._chunks = db

        self._create_rag_block()

    def _create_rag_block(self):
        """Creates the embedding index and encoder for the RAG model."""
        #self.encoder = Embedder("thenlper/gte-large", self._chunks, index_name=f"./data/{self.name}")
        self.encoder = EmbedderOpenAI("text-embedding-ada-002", self._chunks, f"./data/{self.name}", api_key)

    def similarity_search(self, query, n_results=5, return_indexes=False):
        """Performs a similarity search and filters results.

        Args:
            query: The user's query.
            n_results: The maximum number of results to return.
            return_indexes: If True, returns raw indexes. Otherwise, returns text.

        Returns:
            A tuple of filtered distances and results.
        """
        distances, results = self.encoder.similarity_search(query, n_results, return_indexes)
        print("dist",distances)
        print("results", results)
        keep_mask = distances < self.similarity_threshold
        filtered_distances = distances[keep_mask]
        filtered_results = results[keep_mask]

        return filtered_distances, filtered_results
