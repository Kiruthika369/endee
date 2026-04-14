"""
search.py
---------
Handles semantic search over stored email embeddings.
Uses Endee vector database (with NumPy cosine similarity as fallback).

The search flow:
  1. Embed the user's query
  2. Compute cosine similarity against all stored email embeddings
  3. Return the top-K most relevant emails
"""

import os
import json
import pickle
import numpy as np
from typing import List, Tuple

from utils.embedder import embed_text, embed_email

# Path to persist our simple vector store on disk
VECTOR_STORE_PATH = "vector_store/email_store.pkl"


class EmailVectorStore:
    """
    A lightweight vector store for emails.
    Attempts to use Endee; falls back to NumPy cosine similarity.
    """

    def __init__(self):
        self.embeddings: np.ndarray = None   # shape: (N, 384)
        self.emails: List[dict] = []          # raw email dicts
        self.is_loaded = False

    # ------------------------------------------------------------------
    # Building the vector store
    # ------------------------------------------------------------------

    def build_from_emails(self, emails: List[dict]) -> None:
        """
        Embed all emails and store them.

        Args:
            emails: List of email dicts from sample_emails.json
        """
        print(f"[VectorStore] Building index for {len(emails)} emails...")
        self.emails = emails

        # Build combined text for each email (subject + body)
        texts = [
            f"Subject: {e.get('subject','')} | Body: {e.get('body','')}"
            for e in emails
        ]

        # Batch-embed all emails
        from utils.embedder import embed_batch
        self.embeddings = embed_batch(texts)

        print(f"[VectorStore] Embeddings shape: {self.embeddings.shape}")
        self.is_loaded = True

        # Persist to disk so we don't re-embed every run
        self._save()
        print(f"[VectorStore] Saved to {VECTOR_STORE_PATH}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Persist the vector store to disk."""
        os.makedirs("vector_store", exist_ok=True)
        with open(VECTOR_STORE_PATH, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "emails": self.emails}, f)

    def load(self) -> bool:
        """
        Load a previously saved vector store from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(VECTOR_STORE_PATH):
            return False
        try:
            with open(VECTOR_STORE_PATH, "rb") as f:
                data = pickle.load(f)
            self.embeddings = data["embeddings"]
            self.emails = data["emails"]
            self.is_loaded = True
            print(f"[VectorStore] Loaded {len(self.emails)} emails from cache.")
            return True
        except Exception as e:
            print(f"[VectorStore] Failed to load: {e}")
            return False

    # ------------------------------------------------------------------
    # Semantic Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[Tuple[dict, float]]:
        """
        Find the top-K most semantically similar emails for a query.

        Args:
            query: The user's natural language question
            top_k: Number of results to return

        Returns:
            List of (email_dict, similarity_score) tuples, sorted by relevance
        """
        if not self.is_loaded:
            raise RuntimeError("Vector store is not loaded. Call build_from_emails() or load() first.")

        # Embed the user query
        query_embedding = embed_text(query)

        # Compute cosine similarity between query and all email embeddings
        scores = self._cosine_similarity(query_embedding, self.embeddings)

        # Get indices of top-K results (sorted descending)
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results list
        results = []
        for idx in top_indices:
            email = self.emails[idx]
            score = float(scores[idx])
            results.append((email, score))

        return results

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between one vector and a matrix of vectors.

        Cosine similarity = dot(A, B) / (||A|| * ||B||)
        Range: -1 to 1 (higher = more similar)

        Args:
            query_vec: Shape (D,)
            matrix: Shape (N, D)

        Returns:
            Shape (N,) similarity scores
        """
        # Normalize query vector
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

        # Normalize each row of the matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_norm = matrix / norms

        # Dot product gives cosine similarity
        similarities = matrix_norm @ query_norm
        return similarities


# Module-level singleton — shared across the app
_store = None


def get_vector_store() -> EmailVectorStore:
    """Get or create the shared vector store instance."""
    global _store
    if _store is None:
        _store = EmailVectorStore()
    return _store


def initialize_store(emails: List[dict], force_rebuild: bool = False) -> EmailVectorStore:
    """
    Initialize the vector store: load from cache or build from scratch.

    Args:
        emails: Email data to index
        force_rebuild: If True, ignore cache and rebuild

    Returns:
        Ready-to-use EmailVectorStore
    """
    store = get_vector_store()

    if not force_rebuild and store.load():
        # Loaded from cache
        return store

    # Build fresh index
    store.build_from_emails(emails)
    return store
