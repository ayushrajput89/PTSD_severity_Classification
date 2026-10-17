"""Data loading utilities."""
import os
import yaml
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: The configuration dictionary.
    """
    logger.info(f"Loading config from {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw PTSD data from CSV or XLSX formats.
    
    Args:
        filepath (str): Path to the raw data file.
        
    Returns:
        pd.DataFrame: The loaded raw data.
    """
    logger.info(f"Loading raw data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
        
    _, ext = os.path.splitext(filepath)
    if ext.lower() == ".csv":
        df = pd.read_csv(filepath, encoding='latin-1')
    elif ext.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format for raw data: {ext}. Must be .csv or .xlsx")
        
    logger.info(f"Loaded raw data with shape {df.shape}")
    return df

def load_processed_data(filepath: str) -> pd.DataFrame:
    """Loads processed CSV data.
    
    Args:
        filepath (str): Path to the processed CSV file.
        
    Returns:
        pd.DataFrame: The loaded processed data.
    """
    logger.info(f"Loading processed data from {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed data file not found: {filepath}")
        
    df = pd.read_csv(filepath)
    logger.info(f"Loaded processed data with shape {df.shape}")
    return df

def load_splits(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads train, validation, and test splits based on config.
    
    Args:
        config (dict): Configuration dictionary.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df
    """
    train_path = config["data"]["train_file"]
    val_path = config["data"]["val_file"]
    test_path = config["data"]["test_file"]
    
    logger.info("Loading data splits...")
    train_df = load_processed_data(train_path)
    val_df = load_processed_data(val_path)
    test_df = load_processed_data(test_path)
    
    return train_df, val_df, test_df
