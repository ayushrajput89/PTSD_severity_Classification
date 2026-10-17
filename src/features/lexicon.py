"""
Clinical & trauma lexicon features module.
"""
import logging
import re
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

TRAUMA_LEXICON = {
    "trauma", "assault", "attack", "abuse", "violence", "rape", "combat", "war",
    "accident", "injury", "wound", "damage", "harm", "threat", "danger", "shooting",
    "bombing", "explosion", "disaster", "earthquake", "flood", "fire", "stabbing",
    "murder", "kidnap", "torture", "hostile", "invasion", "battlefield", "gunshot",
    "shelling", "massacre", "slaughter", "ambush", "hostage", "captive", "prisoner",
    "victim", "survivor", "witness", "casualty", "fatality", "corpse", "funeral",
    "mourning", "grief", "loss", "bereavement"
}

PTSD_SYMPTOM_LEXICON = {
    "re_experiencing": {
        "flashback", "nightmare", "intrusive", "reliving", "recurring", "triggered"
    },
    "avoidance": {
        "avoid", "numb", "detach", "withdraw", "isolate", "suppress"
    },
    "hyperarousal": {
        "hypervigilant", "startle", "insomnia", "irritable", "agitated", "restless",
        "alert", "jumpy", "on-edge"
    },
    "negative_cognition": {
        "hopeless", "worthless", "guilty", "shame", "blame", "distrust", "estranged", "diminished"
    }
}

DISSOCIATION_LEXICON = {
    "dissociate", "depersonalize", "derealize", "foggy", "unreal", "disconnected", "out-of-body"
}

def compute_lexicon_counts(text: str, lexicon: Set[str]) -> int:
    """
    Counts matches of lexicon words in text.
    
    Args:
        text: Input text string.
        lexicon: Set of lexicon words.
        
    Returns:
        Integer count of matches.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(1 for w in words if w in lexicon)

def compute_lexicon_ratio(text: str, lexicon: Set[str]) -> float:
    """
    Computes ratio of lexicon word matches to total words.
    
    Args:
        text: Input text string.
        lexicon: Set of lexicon words.
        
    Returns:
        Float ratio.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    if word_count == 0:
        return 0.0
    match_count = sum(1 for w in words if w in lexicon)
    return match_count / word_count

def extract_all_lexicon_features(text: str) -> Dict[str, float]:
    """
    Extracts all clinical and trauma lexicon features.
    
    Args:
        text: Input text string.
        
    Returns:
        Dict with all lexicon counts and ratios.
    """
    features = {}
    
    # Trauma Lexicon
    features['trauma_lexicon_count'] = float(compute_lexicon_counts(text, TRAUMA_LEXICON))
    features['trauma_lexicon_ratio'] = compute_lexicon_ratio(text, TRAUMA_LEXICON)
    
    # PTSD Symptom Lexicons
    for cluster, lexicon in PTSD_SYMPTOM_LEXICON.items():
        features[f'ptsd_{cluster}_count'] = float(compute_lexicon_counts(text, lexicon))
        features[f'ptsd_{cluster}_ratio'] = compute_lexicon_ratio(text, lexicon)
        
    # Dissociation Lexicon
    features['dissociation_lexicon_count'] = float(compute_lexicon_counts(text, DISSOCIATION_LEXICON))
    features['dissociation_lexicon_ratio'] = compute_lexicon_ratio(text, DISSOCIATION_LEXICON)
    
    return features

def batch_extract_lexicon(texts: List[str]) -> pd.DataFrame:
    """
    Applies lexicon extraction to a list of texts.
    
    Args:
        texts: List of text strings.
        
    Returns:
        DataFrame containing lexicon features for each text.
    """
    logger.info(f"Extracting lexicon features for {len(texts)} texts")
    all_features = []
    
    for text in tqdm(texts, desc="Lexicon Features"):
        features = extract_all_lexicon_features(text)
        all_features.append(features)
        
    return pd.DataFrame(all_features)
