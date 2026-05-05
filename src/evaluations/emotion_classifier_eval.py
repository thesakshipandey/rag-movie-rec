#!/usr/bin/env python
"""Evaluate fine-tuned RoBERTa emotion classifier.

Metrics:
- Accuracy, F1-score (macro/micro/weighted)
- Per-class precision, recall, F1
- Confusion matrix
- Emotion distribution analysis
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


def convert_to_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


EMOTIONS = ["Joy", "Trust", "Fear", "Anticipation", "Sadness", "Anger", "Surprise", "Disgust"]


def load_roberta_model(model_path: Path, device: str = "cpu"):
    """Load fine-tuned RoBERTa emotion classifier.
    
    Args:
        model_path: Path to saved model directory
        device: Device to load model on
        
    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        model = model.to(device)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        # Try loading from pretrained base
        try:
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-base", 
                num_labels=8
            )
            # Load just the weights
            state_dict = torch.load(model_path / "pytorch_model.bin", map_location=device)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            return model, tokenizer
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise


def load_test_data(prompts_json: Path, split: str = "test") -> tuple:
    """Load test prompts with emotion labels.
    
    Args:
        prompts_json: Path to prompts.json with emotion labels
        split: Which split to load (test/val)
        
    Returns:
        (texts, labels) where labels are emotion names
    """
    with open(prompts_json, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict):
        # Format: {emotion: [texts]}
        texts, labels = [], []
        for emotion, prompt_list in data.items():
            if emotion in EMOTIONS:
                for text in prompt_list:
                    texts.append(text)
                    labels.append(emotion)
    elif isinstance(data, list):
        # Format: [{text: ..., emotion: ..., split: ...}]
        texts, labels = [], []
        for item in data:
            if item.get('split') == split or split == 'all':
                if 'text' in item and 'emotion' in item:
                    texts.append(item['text'])
                    labels.append(item['emotion'])
    
    return texts, labels


def predict_emotions(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    batch_size: int = 32
) -> np.ndarray:
    """Predict emotion labels for texts.
    
    Args:
        model: RoBERTa model
        tokenizer: Tokenizer
        texts: List of text strings
        device: Device
        batch_size: Batch size for inference
        
    Returns:
        Array of predicted label indices
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
    
    return np.array(predictions)


def evaluate_emotion_classifier(
    model_path: Path,
    test_data_path: Path,
    output_dir: Path,
    split: str = "test",
    device: str = "cpu"
) -> Dict[str, Any]:
    """Comprehensive evaluation of emotion classifier.
    
    Args:
        model_path: Path to saved model
        test_data_path: Path to test data (prompts.json)
        output_dir: Where to save results
        split: Which split to evaluate
        device: Device to use
        
    Returns:
        Dictionary with all metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        model, tokenizer = load_roberta_model(model_path, device)
    except Exception as e:
        print(f"Could not load model: {e}")
        return {"error": str(e)}
    
    print(f"Loading test data from {test_data_path}...")
    texts, label_names = load_test_data(test_data_path, split)
    
    if not texts:
        return {"error": "No test data found"}
    
    print(f"Loaded {len(texts)} test samples")
    
    # Convert labels to indices
    label2id = {e: i for i, e in enumerate(EMOTIONS)}
    y_true = np.array([label2id[label] for label in label_names])
    
    print("Running predictions...")
    y_pred = predict_emotions(model, tokenizer, texts, device)
    
    # Compute metrics
    results = {}
    
    # Overall metrics
    results['overall'] = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
        'f1_micro': float(f1_score(y_true, y_pred, average='micro')),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
        'recall_macro': float(recall_score(y_true, y_pred, average='macro')),
        'num_samples': len(texts)
    }
    
    print(f"\nOverall Accuracy: {results['overall']['accuracy']:.4f}")
    print(f"F1 (macro): {results['overall']['f1_macro']:.4f}")
    print(f"F1 (weighted): {results['overall']['f1_weighted']:.4f}")
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, target_names=EMOTIONS, output_dict=True)
    results['per_class'] = {}
    
    for i, emotion in enumerate(EMOTIONS):
        results['per_class'][emotion] = {
            'precision': float(class_report[emotion]['precision']),
            'recall': float(class_report[emotion]['recall']),
            'f1': float(class_report[emotion]['f1-score']),
            'support': int(class_report[emotion]['support'])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    # Save results
    with open(output_dir / "test_metrics.json", 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"\nSaved metrics to {output_dir / 'test_metrics.json'}")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=EMOTIONS, columns=EMOTIONS)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.csv'}")
    
    # Save per-class metrics as CSV
    per_class_df = pd.DataFrame(results['per_class']).T
    per_class_df.to_csv(output_dir / "per_class_metrics.csv")
    print(f"Saved per-class metrics to {output_dir / 'per_class_metrics.csv'}")
    
    # Save classification report
    with open(output_dir / "classification_report.txt", 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=EMOTIONS))
    print(f"Saved classification report to {output_dir / 'classification_report.txt'}")
    
    # Analyze predictions
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    true_dist = pd.Series(y_true).value_counts().sort_index()
    
    dist_df = pd.DataFrame({
        'true': true_dist,
        'predicted': pred_dist
    })
    dist_df.index = [EMOTIONS[i] if i < len(EMOTIONS) else f"Label_{i}" for i in dist_df.index]
    dist_df.to_csv(output_dir / "label_distributions.csv")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier")
    parser.add_argument("--model_path", required=True, help="Path to saved model")
    parser.add_argument("--test_data", required=True, help="Path to test data")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    results = evaluate_emotion_classifier(
        model_path=Path(args.model_path),
        test_data_path=Path(args.test_data),
        output_dir=Path(args.output_dir),
        split=args.split,
        device=args.device
    )
    
    if "error" not in results:
        print("\n=== Evaluation Complete ===")
        print(f"Accuracy: {results['overall']['accuracy']:.4f}")
        print(f"F1 (macro): {results['overall']['f1_macro']:.4f}")
        print(f"F1 (weighted): {results['overall']['f1_weighted']:.4f}")
    else:
        print(f"Evaluation failed: {results['error']}")

