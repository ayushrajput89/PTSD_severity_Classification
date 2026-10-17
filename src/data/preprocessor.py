"""Text preprocessing pipeline."""
import re
import pandas as pd
import logging
import html

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Applies comprehensive text cleaning.
    
    Args:
        text (str): Input text string.
        
    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
        
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags and unescape entities
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove @mentions and #hashtags markers (keep the word)
    text = re.sub(r'[@#](\w+)', r'\1', text)
    
    # Remove special characters but keep apostrophes for contractions
    # This keeps letters, numbers, spaces, and apostrophes
    text = re.sub(r"[^\w\s']", ' ', text)
    
    # Collapse multiple spaces and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Applies text cleaning pipeline to the entire dataframe.
    
    Args:
        df (pd.DataFrame): Raw input dataframe.
        config (dict): Configuration dictionary.
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataframe.
    """
    logger.info("Starting text preprocessing...")
    df = df.copy()
    
    # Combine 'message' and 'description' columns into 'text'
    # Fill NAs with empty string before combining
    msg = df['message'].fillna('')
    desc = df['description'].fillna('')
    df['text'] = msg + " " + desc
    
    # Apply clean_text
    logger.info("Cleaning text column...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove rows with null/empty text
    initial_len = len(df)
    df = df[df['text'].str.strip() != '']
    df = df.dropna(subset=['text'])
    logger.info(f"Removed {initial_len - len(df)} rows with empty text.")
    
    # Remove duplicates
    if config['data'].get('remove_duplicates', True):
        initial_len = len(df)
        df = df.drop_duplicates(subset=['text'])
        logger.info(f"Removed {initial_len - len(df)} duplicate rows.")
        
    # Remove texts shorter than min_text_length
    min_length = config['data'].get('min_text_length', 15)
    initial_len = len(df)
    df = df[df['text'].str.len() >= min_length]
    logger.info(f"Removed {initial_len - len(df)} rows shorter than {min_length} characters.")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df
