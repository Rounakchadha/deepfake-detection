"""
This script provides utility functions for working with Keras models, including:
- Saving and loading models.
- Plotting the model architecture.
"""

import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.utils import plot_model

def save_model(model, file_path):
    """
    Saves a Keras model to a file.

    Args:
        model (tf.keras.Model): The model to save.
        file_path (str): The path to save the model to.
    """
    try:
        model.save(file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(file_path):
    """
    Loads a Keras model from a file.

    Args:
        file_path (str): The path to the model file.

    Returns:
        tf.keras.Model: The loaded model, or None if loading fails.
    """
    try:
        model = keras_load_model(file_path)
        print(f"Model loaded successfully from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def plot_model_architecture(model, file_path='model_architecture.png'):
    """
    Plots the model architecture and saves it to a file.

    Args:
        model (tf.keras.Model): The model to plot.
        file_path (str): The path to save the plot to.
    """
    try:
        plot_model(
            model,
            to_file=file_path,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True
        )
        print(f"Model architecture plotted and saved to {file_path}")
    except Exception as e:
        print(f"Error plotting model architecture: {e}")


if __name__ == '__main__':
    # Example usage:
    from mesonet import build_mesonet

    # 1. Build a model
    model = build_mesonet()

    # 2. Plot the architecture
    plot_model_architecture(model, file_path='../docs/mesonet_architecture.png')

    # 3. Save the model
    save_model(model, '../weights/mesonet_temp.h5')

    # 4. Load the model
    loaded_model = load_model('../weights/mesonet_temp.h5')
    
    if loaded_model:
        loaded_model.summary()
