
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from PIL import Image

# Define paths
input_dir = 'datasets/images/tongue/vitamin_B12/'
output_dir = 'augmented_data/images/tongue/vitamin_B12/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load and augment images
for filename in os.listdir(input_dir):
    img_path = os.path.join(input_dir, filename)
    
    # Check if file is an image
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print(f"Processing {filename}...")
    img = load_img(img_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Generate augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        i += 1
        if i > 20:  # Generate 20 augmented images per original image
            break
        augmented_img = array_to_img(batch[0])
        augmented_img_path = os.path.join(output_dir, f'{filename.split(".")[0]}_aug_{i}.jpg')
        augmented_img.save(augmented_img_path)
        print(f"Saved augmented image {augmented_img_path}")
