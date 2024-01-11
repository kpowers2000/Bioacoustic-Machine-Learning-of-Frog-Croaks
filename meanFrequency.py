import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Directory where your frog sound subfolders are located
data_directory = "/Users/keenanpowers/Desktop/Frog ML/frog_sound_data"

# Define a function to calculate the mean frequency for a sound file
def calculate_mean_frequency(file_path):
    sound, sr = librosa.load(file_path, sr=None)
    stft = np.abs(librosa.stft(sound))
    frequencies = librosa.fft_frequencies(sr=sr)
    mean_frequency = np.sum(stft * frequencies[:, np.newaxis], axis=0) / np.sum(stft, axis=0)
    return np.mean(mean_frequency)

# Initialize empty lists to store file names and their mean frequencies
file_names = []
mean_frequencies = []

# Dictionary to store mean frequencies and standard deviations for each frog species
frog_data = {}

# Iterate through each subfolder (each frog species)
for subfolder_name in os.listdir(data_directory):
    subfolder_path = os.path.join(data_directory, subfolder_name)
    if os.path.isdir(subfolder_path):
        print(f"Processing subfolder: {subfolder_name}")
        frog_number = int(subfolder_name.split(" ")[1])

        # Initialize list to store frequencies within the subfolder
        subfolder_frequencies = []

        # Iterate through each file in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.m4a'):  # Assuming files are in .m4a format
                file_path = os.path.join(subfolder_path, filename)
                mean_frequency = calculate_mean_frequency(file_path)
                subfolder_frequencies.append(mean_frequency)

        # Calculate mean and standard deviation for the current subfolder
        mean_value = np.mean(subfolder_frequencies)
        std_dev = np.std(subfolder_frequencies)

        # Store mean frequency and standard deviation in the dictionary
        frog_data[frog_number] = (mean_value, std_dev)

# Unpack dictionary values into lists for plotting
frog_numbers, mean_std_values = zip(*sorted(frog_data.items()))

# Separate mean values and standard deviations
mean_values, std_values = zip(*mean_std_values)

# Plotting the graph with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(frog_numbers, mean_values, yerr=std_values, fmt='o', color='skyblue', ecolor='black', capsize=5)
plt.title('Mean Frequencies of Frog Sounds with Error Bars')
plt.xlabel('Frog Species Number')
plt.ylabel('Mean Frequency (Hz)')
plt.grid(True)
plt.tight_layout()
plt.show()
