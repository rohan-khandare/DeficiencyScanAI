import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the saved model
model = load_model('multimodal_model2.h5')

# Function to preprocess images
def load_image(image):
    img = load_img(image, target_size=(244, 244))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# Convert 'Yes'/'No' responses
def yes_no_to_num(answer):
    return 1 if answer.lower() == 'yes' else 0

# Streamlit UI setup
st.title("Nutritional Deficiency Assessment")

st.header("Upload Required Images")
skin_image = st.file_uploader("Upload Skin Image", type=["jpg", "jpeg", "png"])
eye_image = st.file_uploader("Upload Eye Image", type=["jpg", "jpeg", "png"])
nail_image = st.file_uploader("Upload Nail Image", type=["jpg", "jpeg", "png"])
tongue_image = st.file_uploader("Upload Tongue Image", type=["jpg", "jpeg", "png"])

st.header("Survey and Blood Data Inputs")

age = st.number_input("Age", min_value=0)
gender = st.selectbox("Gender", ["Male", "Female"])
dietary_supplement = st.radio("Do you take dietary supplements?", ["Yes", "No"])
fruits_veg_intake = st.radio("Do you eat fruits and vegetables regularly?", ["Yes", "No"])
meat_dairy_intake = st.radio("Do you consume meat and dairy regularly?", ["Yes", "No"])
sun_exposure = st.radio("Do you get regular sun exposure?", ["Yes", "No"])
fatigue_present = st.radio("Are you feeling fatigued?", ["Yes", "No"])

hemoglobin = st.number_input("Hemoglobin level (g/dL)")
rbc = st.number_input("RBC count (million cells/µL)")
wbc = st.number_input("WBC count (cells/µL)")
platelets = st.number_input("Platelets count (thousand cells/µL)")
ferritin = st.number_input("Ferritin level (ng/mL)")
cholesterol = st.number_input("Cholesterol level (mg/dL)")
glucose = st.number_input("Glucose level (mg/dL)")

# Display results and recommendations
deficiency_recommendations = {
    'Vitamin_A': {
        'recommendation': 'Include carrots, sweet potatoes, spinach, kale, and liver in your diet.',
        'future_risks': 'Increased risk of night blindness, weakened immune system, and dry skin.'
    },
    'Vitamin_B12': {
        'recommendation': 'Include meats, fish, dairy products, eggs, and fortified cereals. Vegetarians can consider supplements.',
        'future_risks': 'Anemia, nerve damage, memory issues, and fatigue.'
    },
    'Vitamin_D': {
        'recommendation': 'Increase sun exposure, consume fatty fish (salmon, mackerel), fortified milk, and egg yolks.',
        'future_risks': 'Bone disorders, muscle weakness, and cardiovascular risks.'
    },
    'Zinc': {
        'recommendation': 'Include red meat, shellfish, legumes, seeds, and nuts in your diet.',
        'future_risks': 'Impaired immune function, hair loss, and delayed wound healing.'
    }
}

if st.button("Predict Deficiency"):
    if skin_image and eye_image and nail_image and tongue_image:
        # Process image uploads
        skin_img = load_image(skin_image)
        eye_img = load_image(eye_image)
        nail_img = load_image(nail_image)
        tongue_img = load_image(tongue_image)

        # Convert survey inputs
        survey_data = np.array([int(age >= 18), int(gender == "Male"), 
                                yes_no_to_num(dietary_supplement), 
                                yes_no_to_num(fruits_veg_intake), 
                                yes_no_to_num(meat_dairy_intake), 
                                yes_no_to_num(sun_exposure), 
                                yes_no_to_num(fatigue_present)])
        survey_data = np.expand_dims(survey_data, axis=0)

        # Collect blood data
        blood_data = np.array([hemoglobin, rbc, wbc, platelets, ferritin, cholesterol, glucose])
        blood_data = np.expand_dims(blood_data, axis=0)

        # Predict deficiencies
        prediction = model.predict([skin_img, eye_img, nail_img, tongue_img, survey_data, blood_data])
        
        # Display prediction and recommendations
        st.subheader("Deficiency Prediction Results")
        vitamin_labels = ['Vitamin_A', 'Vitamin_B12', 'Vitamin_D', 'Zinc']
        
        for i, label in enumerate(vitamin_labels):
            if prediction[0][i] > 0.5:
                st.write(f"**{label}**: Deficiency detected (Confidence: {prediction[0][i]:.2f})")
                st.write(f"- **Recommendation**: {deficiency_recommendations[label]['recommendation']}")
                st.write(f"- **Future Risks**: {deficiency_recommendations[label]['future_risks']}")
            else:
                st.write(f"{label}: No Deficiency detected (Confidence: {prediction[0][i]:.2f})")
    else:
        st.error("Please upload all required images.")
