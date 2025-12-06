"""
Logistic Regression Model Implementation
Extracted from modeling.ipynb for use in Streamlit application
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, make_scorer
)
from models.utils import calculate_metrics, find_optimal_threshold
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_data(data, target_col='Diagnosis', test_size=0.2, random_state=42):
    """Prepare data for training."""
    # Separate features and target
    X = data.drop([target_col, 'PatientID', 'DoctorInCharge'], axis=1, errors='ignore')
    y = data[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return {
        'X_train': X_train_scaled_df,
        'X_test': X_test_scaled_df,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X_train.columns.tolist()
    }


def train_model(data_dict, hyperparameter_tuning=False, random_state=42):
    """Train logistic regression model."""
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    
    if hyperparameter_tuning:
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # Use recall as the scoring metric
        scorer = make_scorer(recall_score)
        
        # Grid search
        grid_search = GridSearchCV(
            LogisticRegression(random_state=random_state, max_iter=1000, class_weight='balanced'),
            param_grid,
            cv=5,
            scoring=scorer,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        best_params = None
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': data_dict['feature_names'],
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'best_params': best_params
    }




def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Alzheimer', 'Alzheimer'],
                yticklabels=['No Alzheimer', 'Alzheimer'], ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, pad=20)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, model_name="Model"):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curve', fontsize=14, pad=20)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_proba, model_name="Model"):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14, pad=20)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_importance, top_n=10):
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=(6, 5))
    top_features = feature_importance.head(top_n)
    colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
    bars = ax.barh(range(len(top_features)), top_features['Coefficient'], 
                   color=colors, alpha=0.7)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=9)
    ax.set_xlabel('Coefficient Value', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    ax.set_title(f'Top {top_n} Features', fontsize=11, pad=15)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        value = row['Coefficient']
        x_pos = value if abs(value) > 0.05 else (0.1 if value >= 0 else -0.1)
        ax.text(x_pos, i, f'{value:.3f}', va='center', 
               ha='left' if value >= 0 else 'right', fontsize=8)
    
    plt.tight_layout()
    return fig

