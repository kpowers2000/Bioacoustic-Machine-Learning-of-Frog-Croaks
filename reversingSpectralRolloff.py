import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Directory where your frog sound subfolders are located
data_directory = "/Users/keenanpowers/Desktop/Frog ML/frog_sound_data"

# Define a function to calculate the frequency above which a given percentage of energy is contained
def calculate_energy_frequency(y, sr, percentage=0.25):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    total_energy = np.sum(rolloff)
    cumulative_energy = np.cumsum(rolloff)
    idx = np.argmax(cumulative_energy >= percentage * total_energy)
    if idx == 0:  # Handle the case when the threshold is not reached
        return 0  # Or return some default frequency value
    else:
        frequencies = librosa.fft_frequencies(sr=sr)
        return frequencies[min(idx, len(frequencies) - 1)]  # Avoid index out of bounds

# Initialize empty lists to store frequencies and labels
frequencies_list = []
labels_list = []

# Iterate through each subfolder (each frog species)
for subfolder_name in os.listdir(data_directory):
    subfolder_path = os.path.join(data_directory, subfolder_name)
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_name}")
        
        # Iterate through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.m4a'):  # Assuming files are in .m4a format
                file_path = os.path.join(subfolder_path, filename)
                sound, sr = librosa.load(file_path, sr=None)
                frequency = calculate_energy_frequency(sound, sr)
                frequencies_list.append(frequency)
                labels_list.append(subfolder_name)  # Use subfolder_name as label

# Convert lists to numpy arrays
frequencies_array = np.array(frequencies_list).reshape(-1, 1)  # Reshape for machine learning input
labels_array = np.array(labels_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(frequencies_array, labels_array, test_size=0.2, random_state=42)

# Initialize a machine learning model (Random Forest Classifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
