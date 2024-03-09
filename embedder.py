import faiss
import numpy as np
import pickle
import os

from sentence_transformers import SentenceTransformer


class Embedder:
    """Encodes text into embeddings and allows efficient similarity search."""

    def __init__(self, encoder_name, series, index_name):
        """Initializes the Embedder.

        Args:
            encoder_name: The name of the SentenceTransformer model to use.
            series: The list of text data to be embedded.
            index_name: The name for the embedding index.
        """
        self._encoder = SentenceTransformer(encoder_name)
        self._series = series
        self._index = None
        self._index_name = index_name
        self._index_dir = f"{index_name}_index"  
        self._load_index()

    def create_embeddings(self, series, force=False):
        """Creates embeddings for the given text data.

        Args:
            series: The list of text data to embed.
            force: If True, forces re-creation of embeddings even if an index exists.
        """
        if self._index is None or force:
            vectors = self._encode(series)
            self._index = self._create_vector_space(vectors)
            self._save_index()

    def _encode(self, series):
        """Encodes a list of text data into embeddings.

        Args:
            series: The list of text data.

        Returns:
            A numpy array of embeddings.
        """
        return self._encoder.encode(series)

    def _create_vector_space(self, vectors):
        """Creates a Faiss index for embedding search.

        Args:
            vectors: A numpy array of embeddings.

        Returns: 
            A Faiss index.
        """
        vector_dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(vector_dimension)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        return index

    def similarity_search(self, query, k=5, return_indexes=False):
        """Performs a similarity search and returns the most similar text.

        Args:
            query: The query text.
            k: The number of top results to return.
            return_indexes: If True, returns raw indexes. Otherwise, returns text.

        Returns:
            A tuple of distances and the most similar items (texts or indexes).
        """
        assert type(query) is str, "query must be a str"
        query_vector = self._encoder.encode(query)
        query_vector = np.array([query_vector])  # Reshape for Faiss
        faiss.normalize_L2(query_vector)

        distances, indexes = self._index.search(query_vector, k=k)

        if return_indexes:
            return distances[0], indexes[0]
        else:
            return distances[0], self._series[indexes[0]]

    def _save_index(self):
        """Saves the embedding index to disk."""
        os.makedirs(self._index_dir, exist_ok=True)
        faiss_path = os.path.join(self._index_dir, "index.faiss")
        pkl_path = os.path.join(self._index_dir, "index.pkl")

        faiss.write_index(self._index, faiss_path)
        with open(pkl_path, "wb") as f:
            pickle.dump(self, f)

    def _load_index(self):
        """Loads the embedding index from disk (if it exists)."""
        faiss_path = os.path.join(self._index_dir, "index.faiss")
        pkl_path = os.path.join(self._index_dir, "index.pkl")

        if os.path.exists(faiss_path) and os.path.exists(pkl_path):
            self._index = faiss.read_index(faiss_path)
            with open(pkl_path, "rb") as f:
                self = pickle.load(f)
            print("Index loaded from existing files.")
        else:
            print("Index files not found. Creating a new index.")
            self.create_embeddings(self._series)

