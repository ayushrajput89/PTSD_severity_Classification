"""Proxy severity labeling and dataset splitting."""
import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from .loader import load_config, load_raw_data
from .preprocessor import preprocess_dataset

logger = logging.getLogger(__name__)

def compute_severity_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Computes a weighted severity score based on reactions.
    
    Args:
        df (pd.DataFrame): DataFrame with reaction columns.
        config (dict): Configuration dictionary containing weights.
        
    Returns:
        pd.DataFrame: DataFrame with the new 'severity_score' column.
    """
    logger.info("Computing severity scores...")
    df = df.copy()
    
    weights = config['data'].get('severity_weights', {})
    w_angry = weights.get('react_angry', 3.0)
    w_sad = weights.get('react_sad', 2.0)
    w_wow = weights.get('react_wow', 1.0)
    
    df['react_angry'] = pd.to_numeric(df['react_angry'], errors='coerce').fillna(0)
    df['react_sad'] = pd.to_numeric(df['react_sad'], errors='coerce').fillna(0)
    df['react_wow'] = pd.to_numeric(df['react_wow'], errors='coerce').fillna(0)
    
    df['severity_score'] = (
        df['react_angry'] * w_angry + 
        df['react_sad'] * w_sad + 
        df['react_wow'] * w_wow
    )
    
    return df

def assign_severity_class(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Assigns discrete severity classes using quantile-based binning.
    
    Args:
        df (pd.DataFrame): DataFrame with 'severity_score' column.
        config (dict): Configuration dictionary containing quantiles.
        
    Returns:
        pd.DataFrame: DataFrame with 'severity_class' and 'severity_label' columns.
    """
    logger.info("Assigning severity classes...")
    df = df.copy()
    
    quantiles = config['data'].get('quantile_bins', [0.33, 0.66])
    class_names = config['data'].get('class_names', ["Low", "Moderate", "High"])
    
    q1 = df['severity_score'].quantile(quantiles[0])
    q2 = df['severity_score'].quantile(quantiles[1])
    
    conditions = [
        (df['severity_score'] <= q1),
        (df['severity_score'] > q1) & (df['severity_score'] <= q2),
        (df['severity_score'] > q2)
    ]
    
    df['severity_class'] = np.select(conditions, class_names, default=class_names[-1])
    
    # Map to integer labels 0, 1, 2
    label_map = {name: i for i, name in enumerate(class_names)}
    df['severity_label'] = df['severity_class'].map(label_map)
    
    return df

def create_splits(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates stratified train, validation, and test splits and saves them.
    
    Args:
        df (pd.DataFrame): Fully processed and labeled DataFrame.
        config (dict): Configuration dictionary.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
    """
    logger.info("Creating data splits...")
    test_size = config['data'].get('test_size', 0.15)
    val_size = config['data'].get('val_size', 0.15)
    random_state = config['data'].get('random_state', 42)
    stratify = config['data'].get('stratify', True)
    
    train_file = config['data']['train_file']
    val_file = config['data']['val_file']
    test_file = config['data']['test_file']
    
    # Adjust test size for train_test_split to handle test first, then val from remaining
    temp_size = test_size + val_size
    test_frac = test_size / temp_size
    
    stratify_col = df['severity_label'] if stratify else None
    
    # First split off train vs temp(val+test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=temp_size, 
        random_state=random_state, 
        stratify=stratify_col
    )
    
    temp_stratify_col = temp_df['severity_label'] if stratify else None
    
    # Then split temp into val and test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=test_frac, 
        random_state=random_state, 
        stratify=temp_stratify_col
    )
    
    # Save splits
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(val_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    
    logger.info(f"Saving train split to {train_file}")
    train_df.to_csv(train_file, index=False)
    
    logger.info(f"Saving val split to {val_file}")
    val_df.to_csv(val_file, index=False)
    
    logger.info(f"Saving test split to {test_file}")
    test_df.to_csv(test_file, index=False)
    
    return train_df, val_df, test_df

def run_labeling_pipeline(config: dict):
    """Orchestrates the full data labeling and splitting pipeline.
    
    Args:
        config (dict): Configuration dictionary.
    """
    logger.info("Starting labeling pipeline...")
    
    raw_file = config['data']['raw_file']
    processed_file = config['data']['processed_file']
    
    # Load raw data
    raw_df = load_raw_data(raw_file)
    
    # Preprocess
    clean_df = preprocess_dataset(raw_df, config)
    
    # Compute severity scores
    scored_df = compute_severity_score(clean_df, config)
    
    # Assign classes
    labeled_df = assign_severity_class(scored_df, config)
    
    # Save processed dataset
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    logger.info(f"Saving processed dataset to {processed_file}")
    labeled_df.to_csv(processed_file, index=False)
    
    # Create splits
    train_df, val_df, test_df = create_splits(labeled_df, config)
    
    # Print summary statistics
    print("--- Pipeline Summary ---")
    print(f"Total processed samples: {len(labeled_df)}")
    print(f"Class distribution:")
    print(labeled_df['severity_class'].value_counts())
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    print("------------------------")
    
    logger.info("Labeling pipeline complete.")
