# Model Architectures

This document provides a detailed look at the deep learning architectures used in this deepfake detection system. We have implemented two primary models: **XceptionNet** (using transfer learning) and **MesoNet**, a lightweight custom CNN.

The choice of these architectures allows us to compare a deep, powerful, pre-trained model (XceptionNet) with a smaller, more efficient model designed specifically for facial forgery detection (MesoNet).

---

## 1. XceptionNet (Transfer Learning)

XceptionNet is a powerful Convolutional Neural Network (CNN) architecture that is 71 layers deep. It is an evolution of the Inception architecture that replaces standard Inception modules with depthwise separable convolutions.

### Why XceptionNet?

- **High Performance:** XceptionNet has demonstrated state-of-the-art performance on large-scale image classification tasks (like ImageNet). By using transfer learning, we can leverage the rich features it has learned.
- **Efficient Feature Extraction:** Depthwise separable convolutions make the architecture more computationally efficient than standard convolutions, allowing for a deeper network with fewer parameters.
- **Proven in Forgery Detection:** The XceptionNet architecture has been successfully used in many top-performing deepfake detection models.

### Our Implementation: Transfer Learning

We use a pre-trained XceptionNet model (trained on ImageNet) and adapt it for our binary classification task (REAL vs. FAKE).

1.  **Load the Base Model:** We load the XceptionNet architecture with weights pre-trained on ImageNet, excluding the final classification layer (`include_top=False`).
2.  **Freeze the Base:** The convolutional layers of the base model are initially frozen. This prevents the learned ImageNet weights from being destroyed during the early stages of training.
3.  **Add a Custom Head:** We add a new classification head on top of the base model. This head consists of:
    -   `GlobalAveragePooling2D`: To reduce the spatial dimensions of the feature maps to a single vector per feature map.
    -   `Dense(1024, activation='relu')`: A fully connected layer to learn high-level combinations of the features.
    -   `Dropout(0.5)`: A dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 during training.
    -   `Dense(1, activation='sigmoid')`: The final output layer with a sigmoid activation function for binary classification.

4.  **Fine-Tuning:** After initial training, we "unfreeze" the top layers of the base XceptionNet model and continue training with a very low learning rate. This allows the model to fine-tune the pre-trained features specifically for the task of deepfake detection.

### Architecture Diagram

A diagram of the model can be generated using the `plot_model_architecture` function in `utils/visualization.py`. An example is provided below.

![XceptionNet Architecture](xceptionnet_architecture.png)  
*(To generate this, you would run the training script for XceptionNet, which can be adapted to save the plot.)*

---

## 2. MesoNet

MesoNet is a lightweight CNN architecture specifically designed for detecting face forgery in videos. Its name is derived from "mesoscopic," indicating its focus on detecting manipulations at a finer level than macroscopic image properties but broader than pixel-level analysis.

### Why MesoNet?

- **Efficiency:** MesoNet has a very small number of parameters, making it fast to train and suitable for real-time detection, even on a CPU.
- **Designed for Deepfakes:** The architecture was specifically tailored for deepfake detection, focusing on the features that are most indicative of manipulation.
- **Good Baseline:** It serves as an excellent baseline to compare against more complex models like XceptionNet.

### Our Implementation

We have implemented the MesoNet architecture as described in the original paper. The model consists of the following layers:

1.  **Input Layer:** `Input(shape=(256, 256, 3))` for the input image.
2.  **Convolutional Block 1:**
    -   `Conv2D(8, (3, 3), activation='relu')`
    -   `MaxPooling2D((2, 2))`
3.  **Convolutional Block 2:**
    -   `Conv2D(8, (5, 5), activation='relu')`
    -   `MaxPooling2D((2, 2))`
4.  **Convolutional Block 3:**
    -   `Conv2D(16, (5, 5), activation='relu')`
    -   `MaxPooling2D((2, 2))`
5.  **Convolutional Block 4:**
    -   `Conv2D(16, (5, 5), activation='relu')`
    -   `MaxPooling2D((4, 4))`
6.  **Classification Head:**
    -   `Flatten()`: To convert the 2D feature maps into a 1D vector.
    -   `Dropout(0.5)`: For regularization.
    -   `Dense(16, activation='relu')`: A small fully connected layer.
    -   `Dropout(0.5)`: More regularization.
    -   `Dense(1, activation='sigmoid')`: The final binary classification layer.

### Architecture Diagram

The architecture of our MesoNet implementation is visualized below. This was generated using our `plot_model_architecture` utility.

![MesoNet Architecture](mesonet_architecture_viz.png)
