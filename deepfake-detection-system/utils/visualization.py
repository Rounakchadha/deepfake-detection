"""
This script provides functions for visualizing data and model-related information.
It helps in understanding the dataset and the model's behavior.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from models.model_utils import plot_model_architecture as plot_arch

def plot_sample_images(images, labels, num_samples=10):
    """
    Plots a grid of sample images with their labels.

    Args:
        images (numpy.ndarray): A batch of images.
        labels (numpy.ndarray): Corresponding labels for the images.
        num_samples (int): The number of samples to plot.
    """
    plt.figure(figsize=(15, 8))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {'FAKE' if labels[i] == 1 else 'REAL'}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, dataset_name, output_path=None):
    """
    Plots the distribution of classes (REAL vs. FAKE).

    Args:
        labels (numpy.ndarray or list): A list of labels ('REAL' or 'FAKE', or 0 and 1).
        dataset_name (str): The name of the dataset for the plot title.
        output_path (str, optional): Path to save the plot. If None, the plot is shown.
    """
    # Convert numerical labels to string labels if necessary
    str_labels = ['FAKE' if l == 1 else 'REAL' for l in labels]
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x=str_labels, palette=['green', 'red'])
    plt.title(f'Class Distribution in {dataset_name}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Class distribution plot saved to {output_path}")
    else:
        plt.show()

def plot_model_architecture(model, file_path='../docs/model_architecture.png'):
    """
    A convenient wrapper for plotting the model architecture.
    """
    plot_arch(model, file_path)


if __name__ == '__main__':
    # Example usage:
    # 1. Plot sample images
    dummy_images = np.random.randint(0, 256, size=(10, 224, 224, 3), dtype=np.uint8)
    dummy_labels = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0])
    plot_sample_images(dummy_images, dummy_labels)
    
    # 2. Plot class distribution
    dummy_labels_for_dist = np.random.randint(0, 2, size=1000)
    plot_class_distribution(dummy_labels_for_dist, "Dummy Dataset", output_path='../outputs/figures/dummy_class_dist.png')
    
    # 3. Plot model architecture (requires a model)
    from models.mesonet import build_mesonet
    model = build_mesonet()
    plot_model_architecture(model, file_path='../docs/mesonet_architecture_viz.png')

