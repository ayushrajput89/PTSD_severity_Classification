import argparse
import yaml
import logging
from src.data.labeler import run_labeling_pipeline
from src.models.trainer import run_classical_pipeline, run_deep_learning_pipeline
from src.data.loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="PTSD Severity NLP Pipeline")
    parser.add_argument('--step', type=str, choices=['data', 'train_classical', 'train_dl', 'all'], 
                        default='data', help="Pipeline step to run")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="Path to config file")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    if args.step in ['data', 'all']:
        print("\n" + "="*50)
        print(">>> RUNNING DATA PREPROCESSING & LABELING")
        print("="*50)
        run_labeling_pipeline(config)
        
    if args.step in ['train_classical', 'all']:
        print("\n" + "="*50)
        print(">>> RUNNING CLASSICAL ML PIPELINE")
        print("="*50)
        run_classical_pipeline(config)
        
    if args.step in ['train_dl', 'all']:
        print("\n" + "="*50)
        print(">>> RUNNING DEEP LEARNING PIPELINE")
        print("="*50)
        run_deep_learning_pipeline(config)

if __name__ == "__main__":
    main()
