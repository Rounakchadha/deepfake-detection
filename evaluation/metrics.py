from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def calculate_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Calculates standard deepfake detection metrics.
    
    Args:
        y_true (list or np.array): True binary labels (0 = REAL, 1 = FAKE).
        y_pred_prob (list or np.array): Predicted probabilities for the FAKE class.
        threshold (float): Probability threshold to classify as FAKE.
        
    Returns:
        dict: A dictionary of computed metrics.
    """
    y_pred = (np.array(y_pred_prob) > threshold).astype(int)
    y_true = np.array(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        # Handles edge cases where test set only has one class
        roc_auc = 0.0
        
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }

def print_metrics(metrics_dict):
    """Utility to print metrics nicely formatted."""
    print("-" * 30)
    print("Evaluation Metrics:")
    print("-" * 30)
    for k, v in metrics_dict.items():
        print(f"{k:<12}: {v:.4f}")
    print("-" * 30)
