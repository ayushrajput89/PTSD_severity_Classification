"""
Feature fusion pipeline module.
"""
import logging
import os
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .embeddings import generate_embeddings
from .sentiment import extract_all_sentiment_features, batch_extract_sentiment
from .linguistic import extract_all_linguistic_features, batch_extract_linguistic
from .lexicon import extract_all_lexicon_features, batch_extract_lexicon

logger = logging.getLogger(__name__)

def build_features_for_text(text: str, sbert_model: str, config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Builds complete feature vector for a single text.
    
    Args:
        text: Input text string.
        sbert_model: Name of the SBERT model to use.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (SBERT embedding array, Dict of meta features).
    """
    # Extract SBERT embedding
    embedding_array = generate_embeddings([text], sbert_model, batch_size=1)[0]
    
    # Extract meta features
    meta_features = {}
    meta_features.update(extract_all_sentiment_features(text, config))
    meta_features.update(extract_all_linguistic_features(text))
    meta_features.update(extract_all_lexicon_features(text))
    
    return embedding_array, meta_features

def build_feature_matrix(df: pd.DataFrame, config: Dict[str, Any], save: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Builds features for entire DataFrame.
    
    Args:
        df: DataFrame containing 'text' column.
        config: Configuration dictionary.
        save: Whether to save embeddings and features to disk.
        
    Returns:
        Tuple of (Embeddings array, DataFrame of meta features).
    """
    texts = df['text'].tolist()
    sbert_model_name = config.get('sbert_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    batch_size = config.get('embedding_batch_size', 32)
    
    logger.info("Generating SBERT embeddings...")
    embeddings_array = generate_embeddings(texts, sbert_model_name, batch_size=batch_size)
    
    logger.info("Extracting meta features...")
    sentiment_df = batch_extract_sentiment(texts, config)
    linguistic_df = batch_extract_linguistic(texts)
    lexicon_df = batch_extract_lexicon(texts)
    
    features_df = pd.concat([sentiment_df, linguistic_df, lexicon_df], axis=1)
    
    if save:
        output_dir = config.get('features_output_dir', 'data/processed')
        os.makedirs(output_dir, exist_ok=True)
        
        embeddings_path = os.path.join(output_dir, 'embeddings.npy')
        features_path = os.path.join(output_dir, 'meta_features.csv')
        
        logger.info(f"Saving embeddings to {embeddings_path}")
        np.save(embeddings_path, embeddings_array)
        
        logger.info(f"Saving meta features to {features_path}")
        features_df.to_csv(features_path, index=False)
        
    return embeddings_array, features_df

def get_meta_feature_names() -> List[str]:
    """
    Returns a list of all meta feature column names.
    Since meta features are dynamically built, this provides a reference template
    based on the extractors.
    
    Returns:
        List of meta feature names.
    """
    # Create a dummy text to run through the pipeline
    dummy_text = "This is a sample text to extract feature names."
    dummy_config = {
        'nrc_emotions': ['anger', 'fear', 'sadness', 'disgust', 'joy', 'trust', 'anticipation', 'surprise'],
        'empath_categories': ['fear', 'violence', 'sadness', 'anger', 'aggression', 'negative_emotion',
                              'suffering', 'death', 'nervousness', 'pain', 'horror', 'shame', 'grief']
    }
    
    sentiment_feats = extract_all_sentiment_features(dummy_text, dummy_config)
    linguistic_feats = extract_all_linguistic_features(dummy_text)
    lexicon_feats = extract_all_lexicon_features(dummy_text)
    
    all_names = list(sentiment_feats.keys()) + list(linguistic_feats.keys()) + list(lexicon_feats.keys())
    return all_names

def prepare_model_input(embeddings: np.ndarray, meta_features_df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Prepares final X matrix by scaling meta features and concatenating with embeddings.
    
    Args:
        embeddings: Numpy array of SBERT embeddings.
        meta_features_df: DataFrame of meta features.
        scaler: Optional pre-fitted StandardScaler. If None, a new one is fitted.
        
    Returns:
        Tuple of (Final X matrix, Fitted StandardScaler).
    """
    logger.info("Preparing final model input...")
    
    # Fill any NaNs in meta features
    meta_features_df = meta_features_df.fillna(0.0)
    meta_values = meta_features_df.values
    
    if scaler is None:
        scaler = StandardScaler()
        scaled_meta = scaler.fit_transform(meta_values)
    else:
        scaled_meta = scaler.transform(meta_values)
        
    # Concatenate embeddings and scaled meta features
    X = np.hstack([embeddings, scaled_meta])
    
    logger.info(f"Final input shape: {X.shape}")
    return X, scaler
