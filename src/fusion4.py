from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load EfficientNet model
pretrained_model = load_model('D:\IPCV-cp\models\efficientnet_nutritional_deficiency_tpu.h5')

# Freeze all layers in the pretrained model
for layer in pretrained_model.layers:
    layer.trainable = False

# Define input layers for images
input_shape = (244, 244, 3)
skin_input = Input(shape=input_shape)  
eye_input = Input(shape=input_shape)
nail_input = Input(shape=input_shape)
tongue_input = Input(shape=input_shape)

# Extract features from images
skin_features = pretrained_model(skin_input)
eye_features = pretrained_model(eye_input)
nail_features = pretrained_model(nail_input)
tongue_features = pretrained_model(tongue_input)

# Concatenate image features
combined_image_features = Concatenate()([skin_features, eye_features, nail_features, tongue_features])

# Add batch normalization and dropout to regularize
combined_image_features = BatchNormalization()(combined_image_features)
combined_image_features = Dropout(0.4)(combined_image_features)

# Survey and blood inputs
survey_input = Input(shape=(7,))  
blood_input = Input(shape=(7,))  

# Fully connected layers for survey and blood inputs
x_survey = Dense(64, activation='relu')(survey_input)
x_survey = BatchNormalization()(x_survey)
x_survey = Dropout(0.4)(x_survey)
x_survey = Dense(32, activation='relu')(x_survey)

x_blood = Dense(64, activation='relu')(blood_input)
x_blood = BatchNormalization()(x_blood)
x_blood = Dropout(0.4)(x_blood)
x_blood = Dense(32, activation='relu')(x_blood)  

# Combine image features, survey, and blood data
combined_features = Concatenate()([combined_image_features, x_survey, x_blood])

# Add fully connected layers with dropout
x = Dense(256, activation='relu')(combined_features)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

# Output layer for multi-label classification of deficiencies
output = Dense(4, activation='sigmoid')(x)

# Compile model with Adam optimizer and learning rate scheduler
multimodal_model = Model(inputs=[skin_input, eye_input, nail_input, tongue_input, survey_input, blood_input], outputs=output)
optimizer = Adam(learning_rate=0.001)
multimodal_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Load and preprocess the dataset
data = pd.read_csv('D:\IPCV-cp\combined_dataset4.csv')
survey_columns = ['Age', 'Gender', 'Dietary Supplement', 'Fruits & Veg Intake', 'Meat & Dairy Intake', 'Sun Exposure', 'Fatigue Present']
blood_columns = ['Hemoglobin', 'RBC', 'WBC', 'Platelets', 'Ferritin', 'Cholesterol', 'Glucose']
survey_data = data[survey_columns].values
blood_data = data[blood_columns].values

# Load images
def load_images(image_paths):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=(244, 244))
        img = img_to_array(img) / 255.0  # Normalize images
        images.append(img)
    return np.array(images)

skin_data = load_images(data['skin_image'].values)
eye_data = load_images(data['eye_image'].values)
nail_data = load_images(data['nail_image'].values)
tongue_data = load_images(data['tongue_image'].values)
labels = data[['Vitamin_A', 'Vitamin_B12', 'Vitamin_D', 'Zinc']].values

# Check for consistent input sizes
assert len(skin_data) == len(eye_data) == len(nail_data) == len(tongue_data) == len(survey_data) == len(blood_data) == len(labels), "Mismatch in data lengths!"

# Split the dataset into training and validation sets
skin_train, skin_val, eye_train, eye_val, nail_train, nail_val, tongue_train, tongue_val, survey_train, survey_val, blood_train, blood_val, y_train, y_val = train_test_split(
    skin_data, eye_data, nail_data, tongue_data, survey_data, blood_data, labels, test_size=0.2, random_state=42
)

# Add early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
multimodal_model.fit(
    [skin_train, eye_train, nail_train, tongue_train, survey_train, blood_train], y_train,
    validation_data=([skin_val, eye_val, nail_val, tongue_val, survey_val, blood_val], y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
multimodal_model.save('multimodal_model2.h5')





