"""
Linguistic & readability features module.
"""
import logging
import math
import re
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import textstat

logger = logging.getLogger(__name__)

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere", "nor",
    "can't", "won't", "don't", "didn't", "doesn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "shouldn't",
    "wouldn't", "couldn't", "mustn't", "cannot"
}
FIRST_PERSON_WORDS = {"i", "me", "my", "mine", "myself"}

def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Extracts readability indices.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with readability features.
    """
    try:
        return {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'smog_index': textstat.smog_index(text),
            'dale_chall_score': textstat.dale_chall_readability_score(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text)
        }
    except Exception as e:
        logger.warning(f"Error computing readability features: {e}. Returning zeros.")
        return {
            'flesch_reading_ease': 0.0,
            'smog_index': 0.0,
            'dale_chall_score': 0.0,
            'coleman_liau_index': 0.0,
            'automated_readability_index': 0.0
        }

def extract_structural_features(text: str) -> Dict[str, float]:
    """
    Extracts structural text features.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with structural features.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    word_count = len(words)
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    
    avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0
    
    unique_words = len(set(words))
    type_token_ratio = unique_words / word_count if word_count > 0 else 0.0
    
    if unique_words > 1 and word_count > 1:
        lexical_diversity = math.log(unique_words) / math.log(word_count)
    else:
        lexical_diversity = 0.0
        
    return {
        'word_count': float(word_count),
        'sentence_count': float(sentence_count),
        'avg_word_length': float(avg_word_length),
        'type_token_ratio': float(type_token_ratio),
        'lexical_diversity': float(lexical_diversity)
    }

def extract_syntactic_features(text: str) -> Dict[str, float]:
    """
    Extracts syntactic features from text.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with syntactic features.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words) if len(words) > 0 else 1
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    
    first_person_count = sum(1 for w in words if w in FIRST_PERSON_WORDS)
    first_person_ratio = first_person_count / word_count
    
    negation_count = sum(1 for w in words if w in NEGATION_WORDS)
    
    question_count = text.count('?')
    question_ratio = question_count / sentence_count
    
    exclamation_count = text.count('!')
    exclamation_ratio = exclamation_count / sentence_count
    
    past_tense_count = sum(1 for w in words if len(w) > 3 and w.endswith('ed'))
    past_tense_ratio = past_tense_count / word_count
    
    return {
        'first_person_ratio': float(first_person_ratio),
        'negation_count': float(negation_count),
        'question_ratio': float(question_ratio),
        'exclamation_ratio': float(exclamation_ratio),
        'past_tense_ratio': float(past_tense_ratio)
    }

def extract_all_linguistic_features(text: str) -> Dict[str, float]:
    """
    Combines all linguistic features.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with all linguistic features.
    """
    features = {}
    features.update(extract_readability_features(text))
    features.update(extract_structural_features(text))
    features.update(extract_syntactic_features(text))
    return features

def batch_extract_linguistic(texts: List[str]) -> pd.DataFrame:
    """
    Applies linguistic extraction to a list of texts.
    
    Args:
        texts: List of text strings.
        
    Returns:
        DataFrame containing linguistic features for each text.
    """
    logger.info(f"Extracting linguistic features for {len(texts)} texts")
    all_features = []
    
    for text in tqdm(texts, desc="Linguistic Features"):
        features = extract_all_linguistic_features(text)
        all_features.append(features)
        
    return pd.DataFrame(all_features)
