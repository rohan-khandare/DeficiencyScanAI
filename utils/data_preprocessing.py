import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
base_dir = "D:/IPCV-cp/data/images"

# Define ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # 20% for validation
)

# Flow from directory for training data
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical',
    subset='training' # set as training data
)

# Flow from directory for validation data
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(244, 244),
    batch_size=32,
    class_mode='categorical',
    subset='validation' # set as validation data
)

# Check the classes being recognized
print("Class indices:", train_generator.class_indices)

# Print the number of classes detected
print("Number of classes detected:", len(train_generator.class_indices))

# Print sample data to verify
x_batch, y_batch = next(train_generator)
print("Sample image batch shape:", x_batch.shape)
print("Sample label batch shape:", y_batch.shape)

train_generator.save('../data/train_generator.pkl')
validation_generator.save('../data/validation_generator.pkl')