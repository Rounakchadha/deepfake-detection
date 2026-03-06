"""
This script performs cross-dataset validation to evaluate the model's generalization capabilities.
It trains the model on one dataset and evaluates it on a different, unseen dataset.
"""

import os
import sys
import argparse
import tensorflow as tf

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_model
from training.evaluate import evaluate_model

def cross_dataset_validation(
    model_name,
    train_dataset_name,
    train_dataset_path,
    eval_dataset_name,
    eval_dataset_path,
    epochs,
    batch_size,
    image_size,
    num_frames,
    max_videos_train,
    max_videos_eval
):
    """
    Performs cross-dataset validation.

    Args:
        model_name (str): The model architecture to use.
        train_dataset_name (str): The name of the dataset to train on.
        train_dataset_path (str): Path to the training dataset directory.
        eval_dataset_name (str): The name of the dataset to evaluate on.
        eval_dataset_path (str): Path to the evaluation dataset directory.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        image_size (tuple): Image size (width, height).
        num_frames (int): Number of frames to extract from each video.
        max_videos_train (int): Max videos for training.
        max_videos_eval (int): Max videos for evaluation.
    """
    print(f"--- Starting Cross-Dataset Validation ---")
    print(f"Training on: {train_dataset_name}")
    print(f"Evaluating on: {eval_dataset_name}")
    
    # 1. Train the model on the training dataset
    train_model(
        model_name=model_name,
        dataset_name=train_dataset_name,
        dataset_path=train_dataset_path,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        num_frames=num_frames,
        max_videos=max_videos_train
    )

    # 2. Evaluate the trained model on the evaluation dataset
    model_path = f"../weights/{model_name}_{train_dataset_name}_final.h5"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Skipping evaluation.")
        return

    print(f"\n--- Evaluating on {eval_dataset_name} ---")
    evaluate_model(
        model_path=model_path,
        dataset_name=eval_dataset_name,
        dataset_path=eval_dataset_path,
        image_size=image_size,
        num_frames=num_frames,
        max_videos=max_videos_eval
    )
    print("--- Cross-Dataset Validation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform cross-dataset validation.")
    parser.add_argument("--model", type=str, default="MesoNet", choices=["XceptionNet", "MesoNet"], help="The model architecture to use.")
    parser.add_argument("--train-dataset", type=str, default="FaceForensics++", help="The dataset to train on.")
    parser.add_argument("--train-path", type=str, default="../data/FaceForensics++", help="Path to the training dataset directory.")
    parser.add_argument("--eval-dataset", type=str, default="Celeb-DF", help="The dataset to evaluate on.")
    parser.add_argument("--eval-path", type=str, default="../data/Celeb-DF", help="Path to the evaluation dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size (width and height).")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract from each video.")
    parser.add_argument("--max-videos-train", type=int, default=500, help="Max videos for training.")
    parser.add_argument("--max-videos-eval", type=int, default=100, help="Max videos for evaluation.")

    args = parser.parse_args()

    cross_dataset_validation(
        model_name=args.model,
        train_dataset_name=args.train_dataset,
        train_dataset_path=args.train_path,
        eval_dataset_name=args.eval_dataset,
        eval_dataset_path=args.eval_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_frames=args.num_frames,
        max_videos_train=args.max_videos_train,
        max_videos_eval=args.max_videos_eval
    )
