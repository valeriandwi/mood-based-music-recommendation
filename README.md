# Emotion-Based Music Recommendation System

This repository contains a machine learning model for emotion recognition from facial expressions, which is a crucial component of a Mood-Based Music Recommendation system. The system aims to analyze users' emotions in real-time and recommend music that aligns with their current mood.

## Table of Contents

- [Introduction](#introduction)
- [Model Overview](#model-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Emotion recognition is the process of identifying human emotions, typically from facial expressions. This project is part of a larger initiative to create a Mood-Based Music Recommendation system, where the user's mood is detected through their facial expressions, and appropriate music is suggested.

## Model Overview

The model in this repository is designed to recognize different emotions from facial images. It is trained using a Convolutional Neural Network (CNN) to classify emotions into categories such as happy, sad, angry, surprised, etc.

## Dataset

The model is trained on a publicly available dataset that contains facial images labeled with corresponding emotions. The dataset is split into training, validation, and test sets to ensure the model generalizes well to new data.

## Model Architecture

The emotion recognition model uses a CNN architecture, which is well-suited for image processing tasks. The architecture includes:

- **Convolutional Layers**: For feature extraction from the input images.
- **Pooling Layers**: To reduce the spatial dimensions of the feature maps.
- **Fully Connected Layers**: For classification based on the extracted features.
- **Softmax Layer**: To output the probability distribution over the emotion classes.

## Training

The model is trained using supervised learning. The following steps outline the training process:

1. **Preprocessing**: The facial images are preprocessed, including resizing, normalization, and data augmentation.
2. **Model Training**: The model is trained on the training set using a cross-entropy loss function and optimized with the Adam optimizer.
3. **Validation**: The model's performance is validated on the validation set during training.
4. **Testing**: After training, the model is evaluated on the test set to assess its accuracy and generalization.

## Usage

To use the emotion recognition model:

1. Clone the repository:
    ```bash
    git clone https://github.com/valeriandwi/mood-based-music-recommendation.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mood-based-music-recommendation
    ```
3. Run the Jupyter Notebook `Emotion_Recognition.ipynb` to see the model in action or to retrain it with new data.

## Results

The model achieves an accuracy of approximately X% on the test set, with detailed performance metrics such as precision, recall, and F1-score provided in the notebook.

## Future Work

- **Integration with Music Recommendation System**: The emotion recognition model will be integrated into a larger system that recommends music based on the detected emotion.
- **Improving Accuracy**: Further tuning of the model and exploration of different architectures to improve accuracy.
- **Real-Time Emotion Detection**: Implementing real-time emotion detection using a webcam or other video input.

## Installation

Ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow or PyTorch
- OpenCV
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Prerequisites:

---

Spotify Genres : https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/code

Moods : https://www.kaggle.com/datasets/deadskull7/fer2013
