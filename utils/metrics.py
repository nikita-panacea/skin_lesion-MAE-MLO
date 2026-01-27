"""
Evaluation metrics
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix
)


def compute_metrics(predictions, targets, num_classes=8):
    """
    Compute classification metrics
    Args:
        predictions: array of predicted labels
        targets: array of ground truth labels
        num_classes: number of classes
    Returns:
        metrics: dict of metric values
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Overall metrics
    accuracy = accuracy_score(targets, predictions)
    precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }
    
    return metrics


def print_metrics(metrics, class_names=None):
    """
    Print metrics in a formatted way
    Args:
        metrics: dict from compute_metrics
        class_names: list of class names
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision_macro']*100:.2f}%")
    print(f"  Recall:    {metrics['recall_macro']*100:.2f}%")
    print(f"  F1-score:  {metrics['f1_macro']*100:.2f}%")
    
    print(f"\nPer-class Performance:")
    for i in range(len(metrics['precision_per_class'])):
        class_name = class_names[i] if class_names else f"Class {i}"
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]*100:.2f}%")
        print(f"    Recall:    {metrics['recall_per_class'][i]*100:.2f}%")
        print(f"    F1-score:  {metrics['f1_per_class'][i]*100:.2f}%")
    
    print("="*60 + "\n")