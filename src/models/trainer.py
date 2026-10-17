import os
import json
import yaml
import logging
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any
from src.models.classical import get_models, compare_models
from src.models.deep_learning import train_bert_model, evaluate_epoch, PTSDDataset, save_bert_model
from src.models.evaluator import generate_evaluation_report
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Loads configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_classical_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full classical machine learning pipeline."""
    logger.info("Starting classical ML pipeline...")
    
    # Load data (mock loading for now, assumes CSVs exist based on config)
    data_dir = config.get('data_dir', 'data/processed')
    
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    except FileNotFoundError:
        logger.error("Processed data not found. Please run preprocessing first.")
        # Create dummy data for fallback
        train_df = pd.DataFrame({'text': ['example ' * 10] * 100, 'label': np.random.randint(0, 4, 100)})
        test_df = pd.DataFrame({'text': ['example ' * 10] * 20, 'label': np.random.randint(0, 4, 20)})
    
    # Extract features (TF-IDF as example)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=config.get('max_features', 1000))
    
    X_train = vectorizer.fit_transform(train_df['text']).toarray()
    y_train = train_df['label'].values
    
    X_test = vectorizer.transform(test_df['text']).toarray()
    y_test = test_df['label'].values
    
    class_names = config.get('class_names', ['Control', 'Mild', 'Moderate', 'Severe'])
    
    models_dict = get_models()
    results = compare_models(models_dict, X_train, y_train, X_test, y_test, class_names)
    
    # Save results
    output_dir = config.get('output_dir', 'models/classical')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def run_deep_learning_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the full deep learning pipeline."""
    logger.info("Starting Deep Learning pipeline...")
    
    data_dir = config.get('data_dir', 'data/processed')
    
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    except FileNotFoundError:
        logger.error("Processed data not found. Please run preprocessing first.")
        # Create dummy data for fallback
        train_df = pd.DataFrame({'text': ['example ' * 10] * 100, 'label': np.random.randint(0, 4, 100)})
        val_df = pd.DataFrame({'text': ['example ' * 10] * 20, 'label': np.random.randint(0, 4, 20)})
        test_df = pd.DataFrame({'text': ['example ' * 10] * 20, 'label': np.random.randint(0, 4, 20)})
        
    train_texts, train_labels = train_df['text'].tolist(), train_df['label'].tolist()
    val_texts, val_labels = val_df['text'].tolist(), val_df['label'].tolist()
    test_texts, test_labels = test_df['text'].tolist(), test_df['label'].tolist()
    
    dl_config = config.get('deep_learning', {})
    
    model, tokenizer, history = train_bert_model(
        train_texts, train_labels,
        val_texts, val_labels,
        dl_config
    )
    
    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = PTSDDataset(test_texts, test_labels, tokenizer, dl_config.get('max_length', 256))
    test_dataloader = DataLoader(test_dataset, batch_size=dl_config.get('batch_size', 16))
    
    test_loss, test_acc, test_preds, test_probs, test_targets = evaluate_epoch(model, test_dataloader, device)
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'predictions': test_preds,
        'probabilities': test_probs,
        'targets': test_targets,
        'history': history
    }
    
    output_dir = config.get('output_dir', 'models/deep_learning')
    save_bert_model(model, tokenizer, output_dir)
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    return results

def run_full_pipeline(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Runs both classical and DL pipelines and compares them."""
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        config = {}
    
    logger.info("=== Starting Full Pipeline ===")
    classical_results = run_classical_pipeline(config)
    dl_results = run_deep_learning_pipeline(config)
    
    all_results = {
        'classical': classical_results,
        'deep_learning': dl_results
    }
    
    report_dir = config.get('report_dir', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    generate_evaluation_report(all_results, report_dir)
    
    with open(os.path.join(report_dir, 'full_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
        
    logger.info("=== Full Pipeline Completed ===")
    return all_results
