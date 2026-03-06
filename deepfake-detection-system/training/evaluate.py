"""
This script evaluates a trained deepfake detection model on a test dataset.
It calculates various performance metrics and generates a confusion matrix.
"""

import os
import sys
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_dataset
from data.preprocessing import preprocess_for_validation
from models.model_utils import load_model

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plots and saves the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    print(f"ROC curve saved to {output_path}")


def evaluate_model(model_path, dataset_name, dataset_path, image_size, num_frames, max_videos):
    """
    Evaluates a trained deepfake detection model.
    """
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    if model is None:
        return

    # 2. Load the dataset
    print(f"Loading {dataset_name} dataset for evaluation...")
    frames, labels = load_dataset(dataset_name, dataset_path, num_frames=num_frames, frame_size=image_size, max_videos=max_videos)
    if len(frames) == 0:
        print("No frames loaded for evaluation. Exiting.")
        return
        
    # 3. Preprocess the data
    print("Preprocessing data...")
    y_true = np.array([1 if label == 'FAKE' else 0 for label in labels])
    dataset = preprocess_for_validation(frames, labels).batch(32).prefetch(tf.data.AUTOTUNE)

    # 4. Make predictions
    print("Making predictions...")
    y_pred_proba = model.predict(dataset)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 5. Generate and print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=['REAL', 'FAKE'])
    print(report)
    
    # Save the report
    report_output_path = f"../outputs/reports/classification_report_{dataset_name}.txt"
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {report_output_path}")

    # 6. Plot and save confusion matrix
    cm_output_path = f"../outputs/figures/confusion_matrix_{dataset_name}.png"
    plot_confusion_matrix(y_true, y_pred, cm_output_path)
    
    # 7. Plot and save ROC curve
    roc_output_path = f"../outputs/figures/roc_curve_{dataset_name}.png"
    plot_roc_curve(y_true, y_pred_proba, roc_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a deepfake detection model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file (.h5).")
    parser.add_argument("--dataset", type=str, default="Celeb-DF", choices=["FaceForensics++", "Celeb-DF", "DFDC"], help="The dataset to evaluate on.")
    parser.add_argument("--dataset-path", type=str, default="../data/Celeb-DF", help="Path to the dataset directory.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size (width and height).")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract from each video.")
    parser.add_argument("--max-videos", type=int, default=100, help="Maximum number of videos to use for evaluation.")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        image_size=(args.image_size, args.image_size),
        num_frames=args.num_frames,
        max_videos=args.max_videos
    )
