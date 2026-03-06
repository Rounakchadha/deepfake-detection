"""
This script implements the MesoNet architecture, a lightweight CNN for deepfake detection.
MesoNet is designed to be efficient and effective for real-time deepfake detection.

Reference:
"MesoNet: a Compact Facial Video Forgery Detection Network"
https://arxiv.org/abs/1809.00888
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def build_mesonet(input_shape=(256, 256, 3), num_classes=1):
    """
    Builds the MesoNet model.

    Args:
        input_shape (tuple): The shape of the input images.
        num_classes (int): The number of output classes (1 for binary classification).

    Returns:
        tf.keras.Model: The compiled MesoNet model.
    """
    x_in = Input(shape=input_shape)

    # Layer 1
    x = Conv2D(8, (3, 3), padding='same', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Layer 2
    x = Conv2D(8, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Layer 3
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    # Layer 4
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(4, 4), padding='same')(x)

    # Flatten and fully connected layers
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='sigmoid')(x)

    # Create the model
    model = Model(inputs=x_in, outputs=predictions)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == '__main__':
    # Build the model
    model = build_mesonet()

    # Print the model summary
    model.summary()
