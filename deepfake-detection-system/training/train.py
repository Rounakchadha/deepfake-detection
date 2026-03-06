"""
This script trains a deepfake detection model on a given dataset.
It handles data loading, preprocessing, model building, training, and evaluation.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import load_dataset, create_train_val_split
from data.preprocessing import preprocess_for_training, preprocess_for_validation
from models.xception_model import build_xception_model, unfreeze_and_fine_tune
from models.mesonet import build_mesonet
from models.model_utils import save_model

def plot_training_history(history, output_path):
    """Plots the training and validation accuracy and loss."""
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Training history plot saved to {output_path}")

def train_model(model_name, dataset_name, dataset_path, epochs, batch_size, image_size, num_frames, max_videos):
    """
    Trains a deepfake detection model.
    """
    # 1. Load the dataset
    print(f"Loading {dataset_name} dataset...")
    frames, labels = load_dataset(dataset_name, dataset_path, num_frames=num_frames, frame_size=image_size, max_videos=max_videos)
    if len(frames) == 0:
        print("No frames loaded. Exiting.")
        return

    # 2. Create a train-validation split
    X_train, X_val, y_train, y_val = create_train_val_split(frames, labels)
    
    # 3. Preprocess the data
    print("Preprocessing data...")
    train_dataset = preprocess_for_training(X_train, y_train, image_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = preprocess_for_validation(X_val, y_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # 4. Build the model
    print(f"Building {model_name} model...")
    if model_name == 'XceptionNet':
        model = build_xception_model(input_shape=(image_size[0], image_size[1], 3))
    elif model_name == 'MesoNet':
        model = build_mesonet(input_shape=(image_size[0], image_size[1], 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.summary()
    
    # 5. Train the model
    print("Starting model training...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'../weights/{model_name}_{dataset_name}.h5',
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
    )

    # 6. Fine-tune the model (if using XceptionNet)
    if model_name == 'XceptionNet':
        print("Starting fine-tuning...")
        unfreeze_and_fine_tune(model)
        history_fine_tune = model.fit(
            train_dataset,
            epochs=epochs // 2,  # Fine-tune for fewer epochs
            validation_data=val_dataset,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        # You can combine histories if needed
        for key in history.history:
            history.history[key].extend(history_fine_tune.history[key])


    # 7. Save the final model
    model_save_path = f"../weights/{model_name}_{dataset_name}_final.h5"
    save_model(model, model_save_path)
    
    # 8. Plot and save training history
    plot_output_path = f"../outputs/figures/{model_name}_{dataset_name}_training_history.png"
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)
    plot_training_history(history, plot_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a deepfake detection model.")
    parser.add_argument("--model", type=str, default="MesoNet", choices=["XceptionNet", "MesoNet"], help="The model architecture to use.")
    parser.add_argument("--dataset", type=str, default="Celeb-DF", choices=["FaceForensics++", "Celeb-DF", "DFDC"], help="The dataset to train on.")
    parser.add_argument("--dataset-path", type=str, default="../data/Celeb-DF", help="Path to the dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--image-size", type=int, default=256, help="Image size (width and height).")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to extract from each video.")
    parser.add_argument("--max-videos", type=int, default=200, help="Maximum number of videos to use from the dataset (for quick tests).")
    
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_frames=args.num_frames,
        max_videos=args.max_videos
    )
