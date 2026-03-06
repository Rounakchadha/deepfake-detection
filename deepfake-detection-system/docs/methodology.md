# Methodology

This document outlines the systematic approach taken in this project, from data acquisition and preparation to model training, evaluation, and interpretation. Our methodology is designed to ensure robustness, reproducibility, and a clear understanding of the model's performance.

---

## 1. Dataset Strategy

A robust deepfake detection model must be able to generalize to different types of manipulations and data sources. To achieve this, we employ a multi-dataset strategy.

### Datasets Used
- **FaceForensics++:** A large-scale dataset containing videos manipulated by four different methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures). This dataset is crucial for training a model that can recognize a variety of manipulation techniques.
- **Celeb-DF (v2):** A high-quality dataset with more advanced deepfake videos that are less prone to common artifacts. It is an excellent benchmark for evaluating a model's ability to detect more subtle fakes.
- **Deepfake Detection Challenge (DFDC):** A very large and diverse dataset created for a Kaggle competition. We use the preview version, which still provides significant diversity in actors, environments, and manipulation quality.

### Rationale
By training and testing on a combination of these datasets, we expose our models to a wide range of real-world scenarios. This helps prevent the model from overfitting to the artifacts of a single dataset and improves its generalization capabilities.

---

## 2. Data Preprocessing and Augmentation

Raw video data is not suitable for direct input into a CNN. Our preprocessing pipeline is a critical step to prepare the data for the models.

1.  **Face Extraction:** Since deepfakes are face manipulations, our first step is to isolate the face. We use **dlib's frontal face detector** to locate and crop faces from each video frame. A margin is added around the detected face to ensure the entire head and some context are included.
2.  **Frame Sampling:** Processing every frame of a video is computationally expensive and often redundant. We sample a fixed number of frames (`num_frames`) from each video at evenly spaced intervals. This provides a temporal summary of the video without the overhead of processing the full sequence.
3.  **Image Resizing:** All extracted face crops are resized to a uniform dimension (e.g., 256x256 for MesoNet, 299x299 for XceptionNet) to match the model's input layer.
4.  **Normalization:** Pixel values of the images are scaled from the range [0, 255] to [0, 1]. This is a standard practice that helps stabilize and speed up the training process.
5.  **Data Augmentation:** To prevent overfitting and improve the model's ability to generalize, we apply random transformations to the training data in real-time. Our augmentation pipeline includes:
    -   Random Horizontal Flips
    -   Random Rotations
    -   Random Zooming
    -   Random Contrast Adjustments

---

## 3. Training Process

The model training is orchestrated by the `training/train.py` script.

- **Batching:** The preprocessed data is fed into the model in batches. This allows the model to update its weights more frequently and requires less memory than processing the entire dataset at once.
- **Optimizer:** We use the **Adam optimizer**, an efficient and widely used optimization algorithm.
- **Loss Function:** Since this is a binary classification problem (REAL vs. FAKE), we use **binary cross-entropy** as the loss function.
- **Callbacks:** We use Keras callbacks to manage the training process:
    -   `ModelCheckpoint`: Automatically saves the model with the best validation accuracy.
    -   `EarlyStopping`: Halts the training process if the validation loss does not improve for a set number of epochs. This prevents overfitting and saves time.

### Transfer Learning and Fine-Tuning (XceptionNet)
For the XceptionNet model, we follow a two-stage training process:
1.  **Feature Extraction:** We first train only the custom classification head while keeping the pre-trained base model frozen. This allows the new layers to learn to classify features extracted by the powerful XceptionNet base.
2.  **Fine-Tuning:** After the initial training, we unfreeze the top layers of the XceptionNet base and continue training with a much lower learning rate. This fine-tunes the pre-trained weights to make them more specific to the deepfake detection task.

---

## 4. Evaluation Metrics

To evaluate the performance of our models, we use a standard set of classification metrics, which are calculated by the `training/evaluate.py` script.

-   **Accuracy:** The proportion of correct predictions. While simple, it can be misleading on imbalanced datasets.
-   **Precision:** Out of all the "FAKE" predictions, how many were actually fakes? (TP / (TP + FP)). High precision is important to minimize false alarms.
-   **Recall (Sensitivity):** Out of all the actual fakes, how many did the model correctly identify? (TP / (TP + FN)). High recall is crucial to ensure that as many deepfakes as possible are caught.
-   **F1-Score:** The harmonic mean of precision and recall. It provides a single score that balances both metrics.
-   **ROC-AUC Score:** The Area Under the Receiver Operating Characteristic Curve. It measures the model's ability to distinguish between the REAL and FAKE classes across all classification thresholds. An AUC of 1.0 indicates a perfect classifier.

---

## 5. Cross-Dataset Validation

To truly test the generalization of our models, we perform cross-dataset validation using the `training/cross_validation.py` script. This involves:

1.  Training a model on one dataset (e.g., FaceForensics++).
2.  Evaluating its performance on a completely different, unseen dataset (e.g., Celeb-DF).

This is a much more challenging and realistic test of a model's ability to perform "in the wild," where it will encounter fakes created with techniques it has never seen before.

---

## 6. Interpretability with Grad-CAM

It is not enough for a model to be accurate; it is also important to understand *why* it makes the decisions it does. For this, we use **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM produces a heatmap that highlights the regions of the input image that were most influential in the model's prediction. In our system, this allows us to visualize which parts of a face (e.g., eyes, mouth, cheeks) the model is looking at to determine if it is a deepfake. This is a powerful tool for debugging the model and building trust in its predictions. The Grad-CAM visualization is integrated directly into our Streamlit frontend.
