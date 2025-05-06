# import os
# import random
# import pandas as pd
# from sklearn.utils import shuffle

# # Load the CSV data (survey data and blood data)
# text_csv = pd.read_csv('D:/IPCV-cp/data/new_text_data.csv')  # Survey data (7 fields)
# blood_csv = pd.read_csv('D:/IPCV-cp/data/new_blood_data.csv')  # Blood data (7 fields)

# # Folder paths for images (update these to your actual paths)
# image_folders = {
#     'vitamin_A': 'D:/IPCV-cp/data/images/vitamin_A',  # Eye images
#     'vitamin_B12': 'D:/IPCV-cp/data/images/vitamin_B12',  # Tongue images
#     'vitamin_D': 'D:/IPCV-cp/data/images/vitamin_D',  # Skin images
#     'zinc': 'D:/IPCV-cp/data/images/zinc'  # Nail images
# }

# normal_folders = {
#     'normal_eye': 'D:/IPCV-cp/data/images/normaleye',
#     'normal_nail': 'D:/IPCV-cp/data/images/normalnails',
#     'normal_tongue': 'D:/IPCV-cp/data/images/normaltongue',
#     'normal_skin': 'D:/IPCV-cp/data/images/normalskin'
# }

# # Function to sample images from each folder
# def load_sampled_images(folder_path, num_samples):
#     image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpeg')]
    
#     if len(image_paths) < num_samples:
#         print(f"Warning: Not enough images in {folder_path}. Found {len(image_paths)} images, requested {num_samples}.")
#         num_samples = len(image_paths)  # Adjust to available number of images
        
#     sampled_paths = random.sample(image_paths, num_samples)
#     return sampled_paths

# # Load images from each folder (normal and deficient)
# vitamin_A_images = load_sampled_images(image_folders['vitamin_A'], num_samples=335)
# normal_eye_images = load_sampled_images(normal_folders['normal_eye'], num_samples=335)

# vitamin_B12_images = load_sampled_images(image_folders['vitamin_B12'], num_samples=335)
# normal_tongue_images = load_sampled_images(normal_folders['normal_tongue'], num_samples=335)

# vitamin_D_images = load_sampled_images(image_folders['vitamin_D'], num_samples=335)
# normal_skin_images = load_sampled_images(normal_folders['normal_skin'], num_samples=335)

# zinc_images = load_sampled_images(image_folders['zinc'], num_samples=335)
# normal_nail_images = load_sampled_images(normal_folders['normal_nail'], num_samples=335)

# # Shuffle the images (ensure randomness)
# vitamin_A_images = shuffle(vitamin_A_images)
# normal_eye_images = shuffle(normal_eye_images)
# vitamin_B12_images = shuffle(vitamin_B12_images)
# normal_tongue_images = shuffle(normal_tongue_images)
# vitamin_D_images = shuffle(vitamin_D_images)
# normal_skin_images = shuffle(normal_skin_images)
# zinc_images = shuffle(zinc_images)
# normal_nail_images = shuffle(normal_nail_images)

# # Prepare new columns for image paths
# combined_data = pd.concat([text_csv, blood_csv], axis=1)
# combined_data['eye_image'] = ""
# combined_data['nail_image'] = ""
# combined_data['tongue_image'] = ""
# combined_data['skin_image'] = ""

# # Multilabel Y columns: Vitamin_A, Vitamin_B12, Vitamin_D, Zinc (initially all set to 0)
# combined_data['Vitamin_A'] = 0
# combined_data['Vitamin_B12'] = 0
# combined_data['Vitamin_D'] = 0
# combined_data['Zinc'] = 0

# # Function to randomly assign normal or deficiency image
# def randomly_assign_image(normal_image_list, deficiency_image_list, idx):
#     # Decide randomly whether to assign a normal or deficient image
#     if random.choice([True, False]):  # 50% chance for either
#         return deficiency_image_list[idx] if idx < len(deficiency_image_list) else random.choice(deficiency_image_list)
#     else:
#         return normal_image_list[idx] if idx < len(normal_image_list) else random.choice(normal_image_list)

# # Assigning image paths and labels for each row
# def assign_labels(text_row, blood_row, current_idx):
#     labels = {'Vitamin_A': 0, 'Vitamin_B12': 0, 'Vitamin_D': 0, 'Zinc': 0}
    
#     # Randomly assign normal or deficiency images for each part (eye, nail, tongue, skin)
#     combined_data.loc[current_idx, 'eye_image'] = randomly_assign_image(normal_eye_images, vitamin_A_images, current_idx)
#     combined_data.loc[current_idx, 'nail_image'] = randomly_assign_image(normal_nail_images, zinc_images, current_idx)
#     combined_data.loc[current_idx, 'tongue_image'] = randomly_assign_image(normal_tongue_images, vitamin_B12_images, current_idx)
#     combined_data.loc[current_idx, 'skin_image'] = randomly_assign_image(normal_skin_images, vitamin_D_images, current_idx)
    
#     # Image-based deficiency label
#     if 'vitamin_A' in combined_data.loc[current_idx, 'eye_image']:
#         labels['Vitamin_A'] += 1
#     if 'vitamin_B12' in combined_data.loc[current_idx, 'tongue_image']:
#         labels['Vitamin_B12'] += 1
#     if 'vitamin_D' in combined_data.loc[current_idx, 'skin_image']:
#         labels['Vitamin_D'] += 1
#     if 'zinc' in combined_data.loc[current_idx, 'nail_image']:
#         labels['Zinc'] += 1
    
#     # Check text data for deficiencies
#     deficiency = text_row.get('Deficiency')
#     if deficiency == 'Vitamin_A':
#         labels['Vitamin_A'] += 1
#     elif deficiency == 'Vitamin_B12':
#         labels['Vitamin_B12'] += 1
#     elif deficiency == 'Vitamin_D':
#         labels['Vitamin_D'] += 1
#     elif deficiency == 'Zinc':
#         labels['Zinc'] += 1
    
#     # Check blood data for deficiencies
#     deficiency_blood = blood_row.get('Deficiency')
#     if deficiency_blood == 'Vitamin A':
#         labels['Vitamin_A'] += 1
#     elif deficiency_blood == 'Vitamin B12':
#         labels['Vitamin_B12'] += 1
#     elif deficiency_blood == 'Vitamin D':
#         labels['Vitamin_D'] += 1
#     elif deficiency_blood == 'Zinc':
#         labels['Zinc'] += 1
    
#     # Assign 1 if deficiency is found in at least 2 out of 3 places
#     for key in labels.keys():
#         if labels[key] >= 2:
#             combined_data.loc[current_idx, key] = 1

# # Ensure the dataset is capped at 335 rows for each group (total 670 rows)
# num_samples = 335
# combined_data = combined_data[:num_samples]

# # Iterate and combine the dataset row by row, ensuring idx stays within bounds
# for idx in range(min(num_samples, len(combined_data))):
#     # Get text and blood rows (ensure idx is valid)
#     text_row = text_csv.iloc[idx]
#     blood_row = blood_csv.iloc[idx]
    
#     # Assign labels and update the dataset
#     assign_labels(text_row, blood_row, idx)

# # Save the final combined dataset for training
# combined_data.to_csv('D:/IPCV-cp/combined_dataset3.csv', index=False)



















import os
import random
import pandas as pd
from sklearn.utils import shuffle

# Load the CSV data (survey data and blood data)
text_csv = pd.read_csv('D:/IPCV-cp/data/new_text_data.csv')  # Survey data (7 fields)
blood_csv = pd.read_csv('D:/IPCV-cp/data/new_blood_data.csv')  # Blood data (7 fields)

# Folder paths for images (update these to your actual paths)
image_folders = {
    'vitamin_A': 'D:/IPCV-cp/data/images/vitamin_A',  # Eye images
    'vitamin_B12': 'D:/IPCV-cp/data/images/vitamin_B12',  # Tongue images
    'vitamin_D': 'D:/IPCV-cp/data/images/vitamin_D',  # Skin images
    'zinc': 'D:/IPCV-cp/data/images/zinc'  # Nail images
}

normal_folders = {
    'normal_eye': 'D:/IPCV-cp/data/images/normaleye',
    'normal_nail': 'D:/IPCV-cp/data/images/normalnails',
    'normal_tongue': 'D:/IPCV-cp/data/images/normaltongue',
    'normal_skin': 'D:/IPCV-cp/data/images/normalskin'
}

# Function to sample images from each folder
def load_images(folder_path):
    return [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.jpeg')]

# Load images from each folder (normal and deficient)
vitamin_A_images = load_images(image_folders['vitamin_A'])
normal_eye_images = load_images(normal_folders['normal_eye'])

vitamin_B12_images = load_images(image_folders['vitamin_B12'])
normal_tongue_images = load_images(normal_folders['normal_tongue'])

vitamin_D_images = load_images(image_folders['vitamin_D'])
normal_skin_images = load_images(normal_folders['normal_skin'])

zinc_images = load_images(image_folders['zinc'])
normal_nail_images = load_images(normal_folders['normal_nail'])

# Shuffle the images
random.shuffle(vitamin_A_images)
random.shuffle(normal_eye_images)
random.shuffle(vitamin_B12_images)
random.shuffle(normal_tongue_images)
random.shuffle(vitamin_D_images)
random.shuffle(normal_skin_images)
random.shuffle(zinc_images)
random.shuffle(normal_nail_images)

# Prepare new columns for image paths
combined_data = pd.concat([text_csv, blood_csv], axis=1)
combined_data['eye_image'] = ""
combined_data['nail_image'] = ""
combined_data['tongue_image'] = ""
combined_data['skin_image'] = ""

# Multilabel Y columns: Vitamin_A, Vitamin_B12, Vitamin_D, Zinc (initially all set to 0)
combined_data['Vitamin_A'] = 0
combined_data['Vitamin_B12'] = 0
combined_data['Vitamin_D'] = 0
combined_data['Zinc'] = 0

# Assigning image paths and labels for each row
def assign_images_and_labels(row, current_idx):
    # Decide randomly whether to assign a normal or deficiency image for each type
    if random.choice([True, False]):
        combined_data.loc[current_idx, 'eye_image'] = random.choice(vitamin_A_images)
    else:
        combined_data.loc[current_idx, 'eye_image'] = random.choice(normal_eye_images)

    if random.choice([True, False]):
        combined_data.loc[current_idx, 'nail_image'] = random.choice(zinc_images)
    else:
        combined_data.loc[current_idx, 'nail_image'] = random.choice(normal_nail_images)

    if random.choice([True, False]):
        combined_data.loc[current_idx, 'tongue_image'] = random.choice(vitamin_B12_images)
    else:
        combined_data.loc[current_idx, 'tongue_image'] = random.choice(normal_tongue_images)

    if random.choice([True, False]):
        combined_data.loc[current_idx, 'skin_image'] = random.choice(vitamin_D_images)
    else:
        combined_data.loc[current_idx, 'skin_image'] = random.choice(normal_skin_images)

    # Initialize labels for deficiencies
    labels = {'Vitamin_A': 0, 'Vitamin_B12': 0, 'Vitamin_D': 0, 'Zinc': 0}

    # Update labels based on image paths assigned
    if 'vitamin_A' in combined_data.loc[current_idx, 'eye_image']:
        labels['Vitamin_A'] = 1
    if 'vitamin_B12' in combined_data.loc[current_idx, 'tongue_image']:
        labels['Vitamin_B12'] = 1
    if 'vitamin_D' in combined_data.loc[current_idx, 'skin_image']:
        labels['Vitamin_D'] = 1
    if 'zinc' in combined_data.loc[current_idx, 'nail_image']:
        labels['Zinc'] = 1

    # Assign 1 if deficiency is found in the images
    for key in labels.keys():
        combined_data.loc[current_idx, key] = labels[key]

# Ensure the dataset is capped at 335 rows
num_samples = 335
combined_data = combined_data[:num_samples]

# Iterate and combine the dataset row by row
for idx in range(min(num_samples, len(combined_data))):
    assign_images_and_labels(combined_data.iloc[idx], idx)

# Save the final combined dataset for training
combined_data.to_csv('D:/IPCV-cp/combined_dataset4.csv', index=False)
