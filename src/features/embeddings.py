"""
SBERT embedding generation module.
"""
import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

_model_cache = {}

def load_sbert_model(model_name: str) -> SentenceTransformer:
    """
    Loads and caches SentenceTransformer model.
    
    Args:
        model_name: Name of the SBERT model to load.
        
    Returns:
        SentenceTransformer model.
    """
    if model_name not in _model_cache:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def generate_embeddings(texts: List[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """
    Generates SBERT embeddings for a list of texts.
    
    Args:
        texts: List of text strings.
        model_name: Name of the SBERT model to use (e.g. 'sentence-transformers/all-MiniLM-L6-v2').
        batch_size: Batch size for embedding generation.
        
    Returns:
        Numpy array of shape (n_texts, embedding_dim)
    """
    model = load_sbert_model(model_name)
    logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    
    # We use model.encode which already supports batching and tqdm if show_progress_bar is set
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """
    Saves embeddings to a .npy file.
    
    Args:
        embeddings: Numpy array of embeddings.
        filepath: Path to save the .npy file.
    """
    logger.info(f"Saving embeddings to {filepath}")
    np.save(filepath, embeddings)

def load_embeddings(filepath: str) -> np.ndarray:
    """
    Loads embeddings from a .npy file.
    
    Args:
        filepath: Path to the .npy file.
        
    Returns:
        Numpy array of embeddings.
    """
    logger.info(f"Loading embeddings from {filepath}")
    return np.load(filepath)
