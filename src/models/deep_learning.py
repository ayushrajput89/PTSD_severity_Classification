import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class PTSDDataset(Dataset):
    """PyTorch Dataset for PTSD text classification."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: DistilBertTokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item: int):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class PTSDClassifier(nn.Module):
    """DistilBERT based classifier for PTSD severity."""
    def __init__(self, n_classes: int):
        super(PTSDClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # DistilBERT doesn't have pooler_output, we use the first token (CLS)
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        logits = self.out(output)
        return logits, outputs.attentions

    def get_attention_weights(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Returns attention weights for explainability."""
        _, attentions = self.forward(input_ids, attention_mask)
        return attentions

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: Any, 
    device: torch.device
) -> Tuple[float, float]:
    """Trains the model for one epoch."""
    model = model.train()
    
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(input_ids, attention_mask)
        _, preds = torch.max(logits, dim=1)
        
        loss = loss_fn(logits, targets)
        
        correct_predictions += torch.sum(preds == targets)
        total_predictions += targets.size(0)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
    return np.mean(losses), (correct_predictions.double() / total_predictions).item()

def evaluate_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> Tuple[float, float, List[int], List[List[float]], List[int]]:
    """Evaluates the model for one epoch."""
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['label'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            loss = loss_fn(logits, targets)
            
            correct_predictions += torch.sum(preds == targets)
            total_predictions += targets.size(0)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
            
    return np.mean(losses), (correct_predictions.double() / total_predictions).item(), all_preds, all_probs, all_targets

def train_bert_model(
    train_texts: List[str], 
    train_labels: List[int], 
    val_texts: List[str], 
    val_labels: List[int], 
    config: Dict[str, Any]
) -> Tuple[nn.Module, DistilBertTokenizer, Dict[str, List[float]]]:
    """Full training pipeline for DistilBERT."""
    logger.info("Initializing DistilBERT training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Extract config
    max_length = config.get('max_length', 256)
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 4)
    learning_rate = config.get('learning_rate', 2e-5)
    n_classes = config.get('num_classes', 4)
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = PTSDDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = PTSDDataset(val_texts, val_labels, tokenizer, max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = PTSDClassifier(n_classes).to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_accuracy = 0
    best_model_state = None
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        val_loss, val_acc, _, _, _ = evaluate_epoch(model, val_dataloader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_state = model.state_dict().copy()
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, tokenizer, history

def predict_with_bert(
    model: nn.Module, 
    tokenizer: DistilBertTokenizer, 
    texts: List[str], 
    device: torch.device, 
    max_length: int = 256
) -> Tuple[List[int], List[List[float]]]:
    """Batch inference using trained DistilBERT."""
    model = model.eval()
    
    dataset = PTSDDataset(texts, [0]*len(texts), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=16)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            
    return all_preds, all_probs

def save_bert_model(model: nn.Module, tokenizer: DistilBertTokenizer, save_dir: str) -> None:
    """Saves the trained model and tokenizer."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")

def load_bert_model(save_dir: str, device: torch.device, n_classes: int = 4) -> Tuple[nn.Module, DistilBertTokenizer]:
    """Loads a trained model and tokenizer."""
    tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
    model = PTSDClassifier(n_classes)
    model.load_state_dict(torch.load(os.path.join(save_dir, 'model.bin'), map_location=device))
    model = model.to(device)
    logger.info(f"Model loaded from {save_dir}")
    return model, tokenizer
