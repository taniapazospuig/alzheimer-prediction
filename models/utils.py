"""
Shared utility functions for model implementations
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['True Positives'] = tp
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['Sensitivity'] = metrics['Recall']
    
    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        metrics['PR-AUC'] = average_precision_score(y_true, y_proba)
    
    return metrics


def find_optimal_threshold(y_true, y_proba, metric='f1', min_recall=None):
    """Find optimal threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    youden_scores = tpr - fpr
    
    if metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'recall':
        optimal_idx = np.argmax(recall)
    elif metric == 'precision':
        optimal_idx = np.argmax(precision)
    elif metric == 'youden':
        optimal_idx = np.argmax(youden_scores)
    else:
        optimal_idx = np.argmax(f1_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    
    if min_recall is not None:
        valid_indices = np.where(recall >= min_recall)[0]
        if len(valid_indices) > 0:
            valid_f1 = f1_scores[valid_indices]
            best_valid_idx = valid_indices[np.argmax(valid_f1)]
            optimal_threshold = thresholds[best_valid_idx]
            optimal_idx = best_valid_idx
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    metrics = {
        'threshold': optimal_threshold,
        'precision': precision[optimal_idx],
        'recall': recall[optimal_idx],
        'f1': f1_scores[optimal_idx],
        'predictions': y_pred_optimal
    }
    
    return optimal_threshold, metrics

