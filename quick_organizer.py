import os
import shutil

# Path to the folder containing the frog sound files
folder_path = '/Users/keenan/Desktop/Frog ML/frog_sound_data'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.m4a'):
        # Parse the frog species number (XX) from the filename
        frog_species = int(filename.split(' ')[0])

        # Create subfolder names based on frog species
        subfolder_name = f'Frog {frog_species}'

        # Create subfolders if they don't exist
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move the file to the corresponding subfolder
        file_source = os.path.join(folder_path, filename)
        file_destination = os.path.join(subfolder_path, filename)
        shutil.move(file_source, file_destination)
