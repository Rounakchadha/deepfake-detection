# Advanced Deepfake Detection System

This repository contains the source code for a complete deep learning-based deepfake detection system. This project was developed as a major college project with the goal of achieving high accuracy in detecting deepfake videos and images. The system is designed to be production-ready and includes a user-friendly frontend for real-time analysis.

This project is suitable for submission to conferences like IEEE and for academic research in the field of media forensics and deep learning.

## 🎯 Project Overview

The core of this project is a Convolutional Neural Network (CNN) based model that can classify a video or image as either "REAL" or "DEEPFAKE". The system is built with state-of-the-art deep learning techniques and includes features for model interpretability.

### Key Features

- **High-Accuracy Deepfake Detection:** Utilizes advanced CNN architectures like XceptionNet and MesoNet.
- **Support for Images and Videos:** Can analyze both static images and video files.
- **Real-Time Frontend:** A clean and interactive web UI built with Streamlit for easy use.
- **Model Interpretability:** Implements Grad-CAM to visualize which parts of a face the model focuses on for its predictions.
- **Cross-Dataset Validation:** The system is designed for robust evaluation across different public datasets.
- **Optimized for Local Execution:** The code is optimized for running on a CPU or Apple Silicon (MPS), making it accessible for users without high-end GPUs.

## ⚙️ Tech Stack

- **Backend:** Python, TensorFlow, Keras
- **Frontend:** Streamlit
- **Computer Vision:** OpenCV, dlib
- **Scientific Computing:** NumPy, scikit-learn
- **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure

The project is organized into the following directories:

- `data/`: Scripts for downloading, loading, and preprocessing datasets.
- `models/`: Contains the definitions of the deep learning models (XceptionNet, MesoNet).
- `training/`: Scripts for training, evaluating, and performing cross-validation.
- `utils/`: Utility functions for face extraction, Grad-CAM, video processing, etc.
- `frontend/`: The Streamlit application for the user interface.
- `weights/`: Saved model weights after training.
- `outputs/`: Directory for saving figures, reports, and predictions.
- `docs/`: Detailed documentation on the project's architecture, methodology, and results.

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/deepfake-detection-system.git
cd deepfake-detection-system
```

### 2. Set Up the Environment

It is recommended to use a virtual environment to manage the project's dependencies.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

On an Apple Silicon Mac, you may need to install `tensorflow-metal` for MPS acceleration:
```bash
pip install tensorflow-metal
```

### 4. Download the Datasets

The project is designed to work with several public deepfake datasets. Due to access restrictions, you will need to download them manually.

Run the following script for instructions:
```bash
python data/download_datasets.py
```
This will guide you on how to download the **FaceForensics++**, **Celeb-DF**, and **DFDC** datasets. Place the downloaded and extracted datasets into the `data/` directory.

### 5. Train a Model

Before you can use the frontend, you need to train a model. You can choose between `XceptionNet` and `MesoNet`. `MesoNet` is recommended for faster training, especially on a CPU.

To train a `MesoNet` model on the `Celeb-DF` dataset:
```bash
python training/train.py --model MesoNet --dataset Celeb-DF --dataset-path data/Celeb-DF --epochs 15 --batch-size 32
```
The trained model will be saved in the `weights/` directory.

### 6. Run the Frontend Application

Once you have a trained model, you can launch the Streamlit web application.

```bash
streamlit run frontend/app.py
```

This will open a new tab in your web browser with the deepfake detection UI. You can now upload an image or video to test the system.

## 📜 Documentation

For more detailed information about the project, please refer to the documents in the `docs/` directory:

- **[Architecture](docs/architecture.md):** A deep dive into the neural network architectures.
- **[Methodology](docs/methodology.md):** Our approach to data, training, and evaluation.
- **[Results](docs/results.md):** Performance metrics and analysis of the trained models.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
