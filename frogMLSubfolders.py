import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Directory where your frog sound subfolders are located
data_directory = "/Users/keenan/Desktop/Frog ML/frog_sound_data"

# Function to extract features from audio files using librosa
def extract_features(file_path):
    # Load audio file
    sound, sr = librosa.load(file_path, sr=None)
    
    # Extract features 
    # Mel-Frequency Cepstral Coefficients - MFCC)
    features_mfcc = librosa.feature.mfcc(y=sound, sr=sr)
    #Spectral Centroid
    features_centroid = librosa.feature.spectral_centroid(y=sound, sr=sr)
    #Spectral Bandwidth
    features_bandwidth = librosa.feature.spectral_bandwidth(y=sound, sr=sr)
    #Spectral Contrast
    features_contrast = librosa.feature.spectral_contrast(y=sound, sr=sr)
    #Spectral Rolloff
    features_rolloff = librosa.feature.spectral_rolloff(y=sound, sr=sr)
    #Mel-Scaled Spectrogram
    features_mss = librosa.feature.melspectrogram(y=sound, sr=sr)
    #Chroma Frequencies
    features_chroma = librosa.feature.chroma_stft(y=sound, sr=sr)
    #Zero-Crossing Rate
    features_zcr = librosa.feature.zero_crossing_rate(y=sound)




    # Alternatively, you can use other features as needed

    # Return the flattened features
    return np.mean(features, axis=1)

# Initialize empty lists to store features and labels
features_list = []
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
                features = extract_features(file_path)
                features_list.append(features)
                labels_list.append(subfolder_name)  # Use subfolder_name as label

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
print(f"Accuracy: {accuracy * 100:.2f}%")
