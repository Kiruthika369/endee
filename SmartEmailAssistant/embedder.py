"""
embedder.py
-----------
Handles converting email text into vector embeddings using
SentenceTransformers. These embeddings enable semantic similarity search.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Load a lightweight but powerful model for semantic embeddings
# all-MiniLM-L6-v2 is fast, small (~80MB), and gives great results
MODEL_NAME = "all-MiniLM-L6-v2"

# Cache the model so we don't reload it on every call
_model = None


def get_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
        print("[Embedder] Model loaded successfully.")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single string into an embedding vector.

    Args:
        text: The text to embed (query or email content)

    Returns:
        A numpy array representing the embedding (384 dimensions)
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def embed_email(email: dict) -> np.ndarray:
    """
    Convert a full email record into an embedding.
    We combine subject + body for richer semantic representation.

    Args:
        email: Dictionary with keys like 'subject', 'body', 'sender', etc.

    Returns:
        A numpy array embedding for the email
    """
    # Combine subject and body to capture full email context
    combined_text = f"Subject: {email.get('subject', '')} | Body: {email.get('body', '')}"
    return embed_text(combined_text)


def embed_batch(texts: list) -> np.ndarray:
    """
    Embed a list of strings efficiently in a single batch.

    Args:
        texts: List of strings to embed

    Returns:
        2D numpy array of shape (len(texts), embedding_dim)
    """
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings
