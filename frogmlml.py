import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Directory where your frog sound files are located
data_directory = "/Users/keenan/Desktop/Frog ML/frog_sound_data"

# Function to extract features from audio files using librosa
def extract_features(file_path):
    # Load audio file
    sound, sr = librosa.load(file_path, sr=None)
    
    # Extract features (Example: using Mel-frequency cepstral coefficients - MFCC)
    
    #features = librosa.feature.mfcc(y=sound, sr=sr)
    #features = librosa.feature.chroma_stft(y=sound, sr=sr)
    features = librosa.feature.spectral_centroid(y=sound, sr=sr)


    # Return the flattened features
    return np.mean(features, axis=1)

# Initialize empty lists to store features and labels
features_list = []
labels_list = []

# Iterate through each file in the data directory
for filename in os.listdir(data_directory):
    print(filename)
    if filename.endswith('.m4a'):  # Assuming files are in .wav format
        print("SUCCESS")
        file_path = os.path.join(data_directory, filename)
        features = extract_features(file_path)
        print(features)
        # Append features and corresponding label to the lists
        features_list.append(features)
        # Extract label from filename or use an external mapping based on your dataset
        label = filename.split('_')[0]  # Assuming label is before the first underscore
        labels_list.append(label)

# Convert lists to numpy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)
#print the values in each training and testing set
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Initialize a machine learning model (Random Forest Classifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
