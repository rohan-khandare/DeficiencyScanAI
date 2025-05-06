import os
import pandas as pd

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Loop through each folder and assign labels based on folder names
base_dir = "datasets/images"

for body_part in os.listdir(base_dir):
    body_part_path = os.path.join(base_dir, body_part)
    if os.path.isdir(body_part_path):
        for deficiency in os.listdir(body_part_path):
            deficiency_path = os.path.join(body_part_path, deficiency)
            if os.path.isdir(deficiency_path):
                for file in os.listdir(deficiency_path):
                    if file.endswith(".jpg") or file.endswith(".png"):  # Ensure it's an image file
                        file_paths.append(os.path.join(deficiency_path, file))
                        labels.append(deficiency)

# Create a DataFrame with file paths and their corresponding labels
data = pd.DataFrame({
    'file_path': file_paths,
    'label': labels
})

# Optionally, save the DataFrame as a CSV file for later use
data.to_csv('labeled_data.csv', index=False)

# Print the first few rows to verify
print(data.head())
