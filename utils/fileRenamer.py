import os

# Set the directory containing the images
folder_path = 'datasets/images/tongue/vitamin_B12'

# List all files in the directory
files = os.listdir(folder_path)

# Sort files to ensure they are renamed in order
files.sort()

# Initialize counter
counter = 1

# Loop through all the files in the directory
for filename in files:
    # Check if the file is an image
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Create the new filename
        new_name = f'vitamin_B12_{counter}.jpeg'
        
        # Construct full file paths
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        
        # Check if the target filename already exists
        if not os.path.exists(new_file):
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} -> {new_file}')
        else:
            print(f'Skipped renaming {old_file} as {new_name} already exists.')
        
        # Increment the counter
        counter += 1
