"""
This script defines a deepfake detection model based on the XceptionNet architecture.
It uses transfer learning with a pre-trained XceptionNet and adds a custom classification head.
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_xception_model(input_shape=(299, 299, 3), num_classes=1):
    """
    Builds a deepfake detection model using the XceptionNet architecture.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes (1 for binary classification).

    Returns:
        tf.keras.Model: The compiled XceptionNet-based model.
    """
    # 1. Load the pre-trained Xception model
    # We use include_top=False to exclude the original classification layer.
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # 2. Freeze the base model's layers
    # This prevents the pre-trained weights from being updated during training.
    base_model.trainable = False

    # 3. Add a custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Reduces spatial dimensions
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)  # Regularization to prevent overfitting
    predictions = Dense(num_classes, activation='sigmoid')(x)  # Sigmoid for binary classification

    # 4. Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def unfreeze_and_fine_tune(model, num_layers_to_unfreeze=20):
    """
    Unfreezes the top layers of the model for fine-tuning.

    Args:
        model (tf.keras.Model): The model to fine-tune.
        num_layers_to_unfreeze (int): The number of top layers to unfreeze.
    """
    # Unfreeze the top N layers of the base model
    for layer in model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    # Re-compile the model with a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print(f"Unfroze the top {num_layers_to_unfreeze} layers for fine-tuning.")


if __name__ == '__main__':
    # Build the model
    model = build_xception_model()

    # Print the model summary
    model.summary()

    # Example of how to unfreeze layers for fine-tuning
    unfreeze_and_fine_tune(model, num_layers_to_unfreeze=30)
    
    # Print the summary again to see the change in trainable parameters
    model.summary()
