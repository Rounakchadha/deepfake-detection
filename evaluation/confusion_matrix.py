import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import io

def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5, class_names=['REAL', 'FAKE'], save_path=None, as_figure=False):
    """
    Computes and plots the confusion matrix.
    
    Args:
        y_true (list or np.array): True binary labels.
        y_pred_prob (list or np.array): Predicted probabilities.
        threshold (float): Decision threshold.
        class_names (list): Names of the classes for axes.
        save_path (str, optional): Automatically saves plot to this path if provided.
        as_figure (bool): Return the matplotlib figure object.
    """
    y_pred = (np.array(y_pred_prob) > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_ylabel('Actual Content')
    ax.set_xlabel('Predicted Representation')
    ax.set_title('Deepfake Detection Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        
    if as_figure:
        return fig
        
    plt.show()

def get_fp_fn_indices(y_true, y_pred_prob, threshold=0.5):
    """
    Returns lists of indices for False Positives and False Negatives
    useful for manual analysis/debugging.
    """
    y_pred = (np.array(y_pred_prob) > threshold).astype(int)
    y_true = np.array(y_true)
    
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
    
    return list(fp_indices), list(fn_indices)
