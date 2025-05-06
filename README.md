# 🩺🫀 DeficiencyScanAI

**DeficiencyScanAI** is a multimodal deep learning system designed to detect nutritional deficiencies using image analysis, blood report values, and user-input health questionnaires. It leverages computer vision and machine learning to provide personalized dietary insights and health guidance.

---

## 🔍 Features

- 📷 **Image-Based Detection**: Analyzes eye, nail, skin, and tongue images to detect visual signs of deficiencies.  
- 📄 **Blood Report Analysis**: Processes 6–7 common lab report values to predict iron, calcium, or vitamin D deficiency.  
- 💬 **Questionnaire Module**: A yes/no symptom-based questionnaire to assist in deficiency classification.  
- 🧠 **Multimodal Intelligence**: Predicts multiple deficiencies using all three modalities with confidence scores.  
- 📊 **Result Dashboard**: Visual display of detected deficiencies and personalized dietary recommendations.

---

## 🧪 Tech Stack

- **Frontend**: `Streamlit`  
- **Machine Learning Frameworks**:  
  - `TensorFlow`  
  - `scikit-learn`  
  - `EfficientNet` for image classification  
  - `Random Forest` for questionnaire and blood data classification  

- **Multimodal Fusion**:  
  - **Early Fusion**: Combines features from all modalities before feeding into the model.  
  - **Late Fusion** *(also supported)*: Independent models for each modality; final decision made by combining individual outputs.  
    - traning scripts are given in src/ for traning individual model

---

## 📁 Project Structure

```bash
DeficiencyScanAI/
│
├── data/                          # Sample data and scripts
│   ├── images/                    # Image data categorized by classes (e.g., vitaminA, normal_eye, etc.)
│   ├── blood_data.csv             # Sample labeled dataset of blood reports
│   ├── text_data.csv              # Sample labeled dataset from symptom questionnaire
│   ├── combine.py                 # Script to merge all three modalities into one dataset
│   └── questionnaire_model/      # Combined and preprocessed dataset
│
├── models/                        # Save the trained models here 
│   ├── efficientnet_nutritional_deficiency_tpu.h5       # Fine-tuned EfficientNet model (HDF5 format)
│   ├── efficientnet_nutritional_deficiency_tpu.keras    # Fine-tuned model in Keras format
│   ├── best_random_forest_model.pkl                     # Trained model for questionnaire data
│   ├── random_forest_blood_report_model.pkl             # Trained model for blood report data
│   └── multimodal_model2.h5                             # Multimodal (fused) model
│
├── src/                           # Source code for training and testing
│   ├── ui.py                      # Streamlit web app
│   └── ...                        # Training, evaluation, and fusion scripts
│
├── utils/                         # Helper functions and preprocessing scripts
│
├── docs/                          # Diagrams, reports, or supplementary documentation
│
├── requirements.txt               # List of required Python packages
```
