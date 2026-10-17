"""
Sentiment & emotion features module.
"""
import logging
from typing import Dict, Any, List

import pandas as pd
from tqdm import tqdm
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import nrclex

logger = logging.getLogger(__name__)

_vader_analyzer = SentimentIntensityAnalyzer()
_empath_analyzer = Empath()

def extract_vader_features(text: str) -> Dict[str, float]:
    """
    Extracts VADER sentiment features.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with compound, pos, neg, neu scores.
    """
    scores = _vader_analyzer.polarity_scores(text)
    return {
        'vader_compound': scores['compound'],
        'vader_pos': scores['pos'],
        'vader_neg': scores['neg'],
        'vader_neu': scores['neu']
    }

def extract_nrc_features(text: str, emotions_list: List[str]) -> Dict[str, float]:
    """
    Extracts NRC emotion features.
    
    Args:
        text: Input text string.
        emotions_list: List of NRC emotions to extract.
        
    Returns:
        Dict with raw emotion counts for specified emotions.
    """
    nrc_obj = nrclex.NRCLex(text)
    raw_counts = nrc_obj.raw_emotion_scores
    
    features = {}
    for emotion in emotions_list:
        features[f'nrc_{emotion}'] = float(raw_counts.get(emotion, 0.0))
        
    return features

def extract_empath_features(text: str, categories_list: List[str]) -> Dict[str, float]:
    """
    Extracts Empath category features.
    
    Args:
        text: Input text string.
        categories_list: List of Empath categories to extract.
        
    Returns:
        Dict with Empath category scores.
    """
    empath_features = _empath_analyzer.analyze(text, categories=categories_list)
    features = {}
    
    if empath_features is None:
        empath_features = {cat: 0.0 for cat in categories_list}
        
    for category in categories_list:
        features[f'empath_{category}'] = float(empath_features.get(category, 0.0))
        
    return features

def extract_textblob_features(text: str) -> Dict[str, float]:
    """
    Extracts TextBlob sentiment features.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with polarity and subjectivity scores.
    """
    blob = TextBlob(text)
    return {
        'textblob_polarity': blob.sentiment.polarity,
        'textblob_subjectivity': blob.sentiment.subjectivity
    }

def extract_all_sentiment_features(text: str, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Combines all sentiment and emotion features into a single dict.
    
    Args:
        text: Input text string.
        config: Configuration dict containing emotions_list and categories_list.
        
    Returns:
        Dict with all sentiment features.
    """
    emotions_list = config.get('nrc_emotions', [
        'anger', 'fear', 'sadness', 'disgust', 'joy', 'trust', 'anticipation', 'surprise'
    ])
    categories_list = config.get('empath_categories', [
        'fear', 'violence', 'sadness', 'anger', 'aggression', 'negative_emotion',
        'suffering', 'death', 'nervousness', 'pain', 'horror', 'shame', 'grief'
    ])
    
    features = {}
    try:
        features.update(extract_vader_features(text))
        features.update(extract_nrc_features(text, emotions_list))
        features.update(extract_empath_features(text, categories_list))
        features.update(extract_textblob_features(text))
    except Exception as e:
        logger.warning(f"Error extracting sentiment features: {e}. Returning zeros.")
        # Fill with zeros on error
        features.update({f'vader_{k}': 0.0 for k in ['compound', 'pos', 'neg', 'neu']})
        features.update({f'nrc_{emotion}': 0.0 for emotion in emotions_list})
        features.update({f'empath_{category}': 0.0 for category in categories_list})
        features.update({'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0})
        
    return features

def batch_extract_sentiment(texts: List[str], config: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies sentiment extraction to a list of texts and returns a DataFrame.
    
    Args:
        texts: List of text strings.
        config: Configuration dictionary.
        
    Returns:
        DataFrame containing sentiment features for each text.
    """
    logger.info(f"Extracting sentiment features for {len(texts)} texts")
    all_features = []
    
    for text in tqdm(texts, desc="Sentiment Features"):
        features = extract_all_sentiment_features(text, config)
        all_features.append(features)
        
    return pd.DataFrame(all_features)
