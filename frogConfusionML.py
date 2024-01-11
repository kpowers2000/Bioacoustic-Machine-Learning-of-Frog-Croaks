"""
File: frogConfusionML.py
Author: Keenan Powers
Email: keenan.f.powers27@gmail.com
Github: TODO
Description:This Python script demonstrates the use of various audio feature extraction methods and machine learning techniques for frog species classification.
It utilizes the 'librosa' library to extract audio features like MFCC, Spectral Centroid, Spectral Bandwidth, Spectral Contrast, Spectral Rolloff, Mel-Scaled Spectrogram, and Chroma Frequencies.
The code employs a Random Forest Classifier from the 'sklearn' library to train on the extracted features and predict frog species based on audio characteristics.
Evaluation metrics such as accuracy scores and confusion matrices are computed to assess the performance of each feature extraction method in classification.
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm

# Directory where your frog sound subfolders are located
data_directory = "/Users/keenanpowers/Desktop/Frog ML/frog_sound_data"

# Define feature extraction methods
feature_methods = {
    "MFCC": librosa.feature.mfcc,
    "Spectral Centroid": librosa.feature.spectral_centroid,
    "Spectral Bandwidth": librosa.feature.spectral_bandwidth,
    "Spectral Contrast": librosa.feature.spectral_contrast,
    "Spectral Rolloff": librosa.feature.spectral_rolloff,
    "Mel-Scaled Spectrogram": librosa.feature.melspectrogram,
    "Chroma Frequencies": librosa.feature.chroma_stft,
}

# Initialize empty dictionaries to store accuracy and confusion matrix for each method
accuracies = {}
confusion_matrices = {}

# Iterate through each feature extraction method
for method_name, method_func in tqdm(feature_methods.items(), desc="Processing methods"):
    print(f"Processing method: {method_name}")
    # Initialize empty lists to store features and labels
    features_list = []
    labels_list = []

    # Iterate through each subfolder (each frog species)
    for subfolder_name in os.listdir(data_directory):
        subfolder_path = os.path.join(data_directory, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Iterate through each file in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.m4a'):
                    file_path = os.path.join(subfolder_path, filename)
                    sound, sr = librosa.load(file_path, sr=None)
                    features = np.mean(method_func(y=sound, sr=sr), axis=1)
                    features_list.append(features)
                    labels_list.append(subfolder_name)

    # Convert lists to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

    # Initialize a machine learning model (Random Forest Classifier as an example)
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    accuracies[method_name] = accuracy

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_test, predictions)
    confusion_matrices[method_name] = conf_matrix

    #Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {method_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Display accuracies for each feature extraction method
for method, acc in accuracies.items():
    print(f"Accuracy for {method}: {acc * 100:.2f}%")
