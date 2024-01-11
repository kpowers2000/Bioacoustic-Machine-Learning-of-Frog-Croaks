import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Path to your frog sound data folder
data_directory = "/Users/keenan/Desktop/Frog ML/frog_sound_data"

# Function to extract features from audio files using Librosa
def extract_features(file_path):
    # Load audio file
    sound, sr = librosa.load(file_path, sr=None)
    
    # Extract features (Example: using Mel-frequency cepstral coefficients - MFCC)
    features = librosa.feature.mfcc(y=sound, sr=sr)
    
    # Return the flattened features
    return np.mean(features, axis=1)

# Load data and extract features
features_list = []
labels_list = []
count = 0
for filename in os.listdir(data_directory):
    if filename.endswith('.m4a'):
        count += 1
        print("success " + str(count)) #this is just to see how many files are being processed
        file_path = os.path.join(data_directory, filename)
        print("file_path " + str(file_path))
        features = extract_features(file_path)
        # Append features and corresponding label to the lists
        features_list.append(features)
        # Extract label from filename or use an external mapping based on your dataset
        label = filename.split('_')[0]  # Assuming label is before the first underscore
        labels_list.append(label)
        
# Convert features and labels to numpy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels_array)
print("features array: " + str(features_array))
print("encoded labels: " + str(encoded_labels))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_array, test_size=0.2, random_state=42)

# Define parameters for Grid Search
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Initialize Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Perform Grid Search for best parameters
grid_search.fit(X_train, y_train)

# Get best parameters and model
best_params = grid_search.best_params_
best_rf_classifier = grid_search.best_estimator_

# Make predictions on the test set using the best model
predictions = best_rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Best Parameters: {best_params}")
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
