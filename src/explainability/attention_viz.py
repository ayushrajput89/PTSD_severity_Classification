import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

def get_attention_weights(model: Any, tokenizer: Any, text: str, device: Any, max_length: int = 256) -> Tuple[torch.Tensor, List[str]]:
    """Extracts attention weights from DistilBERT."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        raise ValueError("Model did not return attentions. Make sure output_attentions=True.")
        
    stacked_attentions = torch.stack(outputs.attentions)
    attention_weights = stacked_attentions.squeeze(1).cpu()
    
    return attention_weights, tokens

def compute_token_importance(attention_weights: torch.Tensor, layer: int = -1) -> np.ndarray:
    """Aggregates attention across heads for a given layer."""
    layer_attention = attention_weights[layer]
    avg_head_attention = torch.mean(layer_attention, dim=0)
    cls_attention = avg_head_attention[0, :].numpy()
    
    return cls_attention

def plot_attention_heatmap(tokens: List[str], attention_weights: torch.Tensor, layer: int = 0, head: int = 0, save_path: Optional[str] = None) -> None:
    """Heatmap of attention matrix."""
    matrix = attention_weights[layer, head].numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
    plt.title(f'Attention Heatmap (Layer {layer}, Head {head})')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_token_importance(tokens: List[str], importance_scores: np.ndarray, prediction: str, save_path: Optional[str] = None) -> None:
    """Bar chart of token importance."""
    plt.figure(figsize=(12, 6))
    
    clean_tokens = []
    clean_scores = []
    for t, s in zip(tokens, importance_scores):
        if t not in ['[CLS]', '[SEP]', '[PAD]']:
            clean_tokens.append(t)
            clean_scores.append(s)
            
    if not clean_tokens:
        clean_tokens = tokens
        clean_scores = importance_scores

    plt.bar(range(len(clean_tokens)), clean_scores)
    plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=90)
    plt.title(f'Token Importance (Prediction: {prediction})')
    plt.xlabel('Tokens')
    plt.ylabel('Attention Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def create_highlighted_html(tokens: List[str], importance_scores: np.ndarray, prediction: str, confidence: float) -> str:
    """Creates HTML with color-coded token highlights."""
    max_score = np.max(importance_scores) if np.max(importance_scores) > 0 else 1.0
    normalized_scores = importance_scores / max_score
    
    html = f"<div><h3>Prediction: {prediction} (Confidence: {confidence:.2f})</h3><p>"
    
    for token, score in zip(tokens, normalized_scores):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        display_token = token.replace('##', '') if token.startswith('##') else f" {token}"
        alpha = float(score)
        bg_color = f"rgba(255, 0, 0, {alpha})"
        
        html += f'<span style="background-color: {bg_color}; padding: 2px; border-radius: 3px;">{display_token}</span>'
        
    html += "</p></div>"
    return html

def explain_with_attention(model: Any, tokenizer: Any, text: str, device: Any, class_names: List[str]) -> Dict[str, Any]:
    """Full attention-based explanation."""
    attention_weights, tokens = get_attention_weights(model, tokenizer, text, device)
    importance_scores = compute_token_importance(attention_weights, layer=-1)
    
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        
    pred_idx = int(np.argmax(probs))
    prediction = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    confidence = float(probs[pred_idx])
    
    html_highlight = create_highlighted_html(tokens, importance_scores, prediction, confidence)
    
    return {
        'tokens': tokens,
        'importance_scores': importance_scores.tolist(),
        'highlighted_html': html_highlight,
        'prediction': prediction,
        'confidence': confidence
    }
