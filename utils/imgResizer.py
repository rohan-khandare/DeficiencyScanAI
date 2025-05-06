from PIL import Image
import os

# Set the path to your images folder
folder_path ='datasets/images/tongue/vitamin_B12'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Full path to the image file
    img_path = os.path.join(folder_path, filename)

    # Open an image file
    with Image.open(img_path) as img:
        # Resize the image
        img_resized = img.resize((244, 244))

        # Save the image back to the same path, overwriting the original file
        img_resized.save(img_path)

print("All images have been resized to 244x244.")
