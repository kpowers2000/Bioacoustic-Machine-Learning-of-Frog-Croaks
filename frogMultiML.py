import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Directory where your frog sound subfolders are located
data_directory = "/Users/keenan/Desktop/Frog ML/frog_sound_data"

# Define feature extraction methods
feature_methods = {
    "MFCC": librosa.feature.mfcc,
    "Spectral Centroid": librosa.feature.spectral_centroid,
    "Spectral Bandwidth": librosa.feature.spectral_bandwidth,
    "Spectral Contrast": librosa.feature.spectral_contrast,
    "Spectral Rolloff": librosa.feature.spectral_rolloff,
    "Mel-Scaled Spectrogram": librosa.feature.melspectrogram,
    "Chroma Frequencies": librosa.feature.chroma_stft,
    "Zero-Crossing Rate": librosa.feature.zero_crossing_rate,
}

# Initialize empty dictionaries to store accuracy for each method
accuracies = {}

# Iterate through each feature extraction method
for method_name, method_func in feature_methods.items():
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

# Display accuracies for each feature extraction method
for method, acc in accuracies.items():
    print(f"Accuracy for {method}: {acc * 100:.2f}%")
