import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the image model (EfficientNet)
efficientnet_model = tf.keras.models.load_model('D:/IPCV-cp/models/efficientnet_nutritional_deficiency_tpu.h5')

# Load the random forest models for symptom data and blood report
with open('random_forest_model.pkl', 'rb') as symptom_model_file:
    symptom_model = pickle.load(symptom_model_file)
with open('blood_model3.pkl', 'rb') as blood_report_model_file:
    blood_report_model = pickle.load(blood_report_model_file)

# Label encoders for symptom data
data = pd.read_csv('nutritional_deficiency_dataset.csv')
symptom_label_encoders = {}
symptom_columns = ['Fatigue', 'Headaches', 'Cramps', 'Veg_Fruit_Intake', 'Skin_Problems', 'Sleep_Hours',
                   'Joint_Pain', 'Hair_Loss', 'Mood_Swings', 'Digestive_Issues', 'Exercise_Frequency',
                   'Sun_Exposure', 'Thirst_Frequency']

for column in symptom_columns:
    le = LabelEncoder()
    le.fit(data[column])
    symptom_label_encoders[column] = le

# Class labels for the EfficientNet model
class_labels = {0: 'normaltongue', 1: 'vitamin_A', 2: 'vitamin_D', 3: 'normalskin',
                4: 'normalnails', 5: 'normaleye', 6: 'vitamin_B12', 7: 'zinc'}

# Deficiency mappings for blood report model
blood_report_deficiency_mapping = {
    0: 'Iron Deficiency',
    1: 'vitamin_D',
    2: 'vitamin_B12',
    3: 'None'
}

# Preprocessing function for images (for EfficientNet)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(244, 244))  # Resize image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Predict deficiency from the image model
def predict_image_deficiency(img_path):
    processed_image = preprocess_image(img_path)
    prediction = efficientnet_model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    return class_labels[predicted_class]

# Predict deficiency from symptom data
def predict_symptom_deficiency(user_symptom_data):
    user_symptom_df = pd.DataFrame([user_symptom_data])
    prediction = symptom_model.predict(user_symptom_df)
    predicted_class = class_labels[prediction[0]]
    return predicted_class

# Predict deficiency from blood report data
def predict_blood_report_deficiency(user_blood_data):
    user_blood_df = pd.DataFrame([user_blood_data])
    scaler = StandardScaler()
    user_blood_scaled = scaler.fit_transform(user_blood_df)
    prediction = blood_report_model.predict(user_blood_scaled)
    predicted_deficiency = blood_report_deficiency_mapping[prediction[0]]
    return predicted_deficiency

# Function to get user input and combine results from all models
def get_combined_assessment():
    # Step 1: Get image input and make prediction
    img_path = input("Enter the path to the image (eye, skin, etc.): ")
    img_prediction = predict_image_deficiency(img_path)
    print(img_prediction)
    # Step 2: Get user input for symptom data
    user_symptom_data = {}
    for column in symptom_columns:
        valid_input = False
        while not valid_input:
            user_input = input(f"Enter value for {column} ({symptom_label_encoders[column].classes_}): ").strip().lower()
            normalized_input = user_input.capitalize()
            if normalized_input in symptom_label_encoders[column].classes_:
                user_symptom_data[column] = symptom_label_encoders[column].transform([normalized_input])[0]
                valid_input = True
            else:
                print(f"Invalid input! Please enter one of the following: {symptom_label_encoders[column].classes_}")
    symptom_prediction = predict_symptom_deficiency(user_symptom_data)
    
    # Step 3: Get user input for blood report data
    blood_report_columns = ['Hemoglobin', 'RBC', 'WBC', 'Platelets', 'Ferritin', 'Cholesterol', 'Glucose']
    user_blood_data = {}
    for column in blood_report_columns:
        valid_input = False
        while not valid_input:
            try:
                user_input = float(input(f"Enter value for {column}: "))
                user_blood_data[column] = user_input
                valid_input = True
            except ValueError:
                print("Invalid input! Please enter a numeric value.")
    blood_report_prediction = predict_blood_report_deficiency(user_blood_data)
    
    # Step 4: Combine results and filter out non-deficiency labels
    combined_results = {
        "Image Model": img_prediction,
        "Symptom Model": symptom_prediction,
        "Blood Report Model": blood_report_prediction
    }
    
    # Exclude non-deficiency results
    non_deficiencies = ['normaltongue', 'normalskin', 'normaleye', 'normalnails', 'None']
    detected_deficiencies = {model: result for model, result in combined_results.items() if result not in non_deficiencies}

    # Display combined results
    print("\n--- Combined Deficiency Assessment ---")
    print(detected_deficiencies)
    
    # Provide recommendations and future risks for detected deficiencies
    if detected_deficiencies:
        for deficiency in detected_deficiencies.values():
            if deficiency in deficiency_recommendations:
                print(f"\nDeficiency Detected: {deficiency}")
                print(f"Dietary Recommendation: {deficiency_recommendations[deficiency]['recommendation']}")
                print(f"Future Risks: {deficiency_recommendations[deficiency]['future_risks']}")
    else:
        print("No significant deficiencies detected.")

# Mapping deficiencies to dietary recommendations and future risks
deficiency_recommendations = {
    'vitamin_A': {
        'recommendation': 'Include carrots, sweet potatoes, spinach, kale, and liver in your diet.',
        'future_risks': 'Increased risk of night blindness, weakened immune system, and dry skin.'
    },
    'vitamin_B12': {
        'recommendation': 'Include meats, fish, dairy products, eggs, and fortified cereals. Vegetarians can consider supplements.',
        'future_risks': 'Anemia, nerve damage, memory issues, and fatigue.'
    },
    'vitamin_D': {
        'recommendation': 'Increase sun exposure, consume fatty fish (salmon, mackerel), fortified milk, and egg yolks.',
        'future_risks': 'Bone disorders, muscle weakness, and cardiovascular risks.'
    },
    'zinc': {
        'recommendation': 'Include red meat, shellfish, legumes, seeds, and nuts in your diet.',
        'future_risks': 'Impaired immune function, hair loss, and delayed wound healing.'
    },
    'Iron Deficiency': {
        'recommendation': 'Include red meat, beans, lentils, spinach, and fortified cereals.',
        'future_risks': 'Anemia, fatigue, weakened immune system, and poor concentration.'
    }
}

# Run the assessment
get_combined_assessment()
