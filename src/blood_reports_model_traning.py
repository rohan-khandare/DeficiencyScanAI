# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Create the DataFrame
# data = {
#     'Hemoglobin': [13.5, 15.2, 10.0, 12.5, 16.0, 9.0, 13.8, 11.5, 14.0, 12.0, 13.0, 11.0, 12.8, 14.5, 9.8, 12.7, 13.2, 15.0, 10.5, 14.2, 12.2, 11.8, 13.6, 15.5, 10.2, 11.6],
#     'RBC': [4.7, 5.1, 3.8, 4.2, 5.6, 3.2, 4.8, 4.0, 4.9, 3.9, 4.5, 3.6, 4.3, 5.2, 3.4, 4.1, 4.7, 5.3, 3.9, 5.0, 4.1, 3.7, 4.6, 5.4, 3.5, 4.0],
#     'WBC': [6.0, 5.5, 8.0, 7.0, 4.5, 10.0, 5.0, 9.5, 5.8, 8.5, 6.2, 9.2, 7.5, 5.0, 9.8, 6.8, 6.0, 4.8, 8.8, 5.5, 7.2, 8.0, 6.3, 4.6, 9.0, 9.2],
#     'Platelets': [250, 280, 180, 270, 320, 150, 290, 220, 275, 200, 260, 180, 240, 310, 170, 230, 290, 300, 190, 295, 240, 220, 270, 310, 170, 210],
#     'Ferritin': [120, 200, 30, 50, 300, 10, 100, 40, 150, 35, 60, 25, 45, 180, 12, 55, 95, 220, 20, 140, 50, 35, 110, 250, 18, 40],
#     'Cholesterol': [180, 190, 220, 240, 150, 260, 200, 210, 175, 230, 190, 250, 210, 160, 270, 200, 180, 170, 240, 180, 210, 220, 190, 160, 260, 215],
#     'Glucose': [90, 100, 160, 140, 110, 180, 95, 150, 100, 170, 115, 160, 140, 90, 190, 145, 100, 105, 175, 90, 135, 155, 100, 95, 185, 150],
#     'Deficiency': ['Iron Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'None', 'Iron Deficiency', 'None', 'Vitamin B12 Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'Iron Deficiency', 'Vitamin B12 Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'None', 'None', 'Iron Deficiency', 'None', 'Vitamin D Deficiency', 'Iron Deficiency', 'None', 'None', 'Iron Deficiency', 'Vitamin B12 Deficiency']
# }

# df = pd.DataFrame(data)

# # Convert categorical 'Deficiency' column to numeric labels
# df['Deficiency'] = df['Deficiency'].astype('category').cat.codes

# # Define features (X) and target (y)
# X = df.drop(columns=['Deficiency'])
# y = df['Deficiency']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Use SMOTE to handle class imbalance
# smote = SMOTE(random_state=42, k_neighbors=2)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Scale the features
# scaler = StandardScaler()
# X_train_res = scaler.fit_transform(X_train_res)
# X_test = scaler.transform(X_test)

# # Initialize and train the RandomForestClassifier
# rf = RandomForestClassifier(random_state=42)
# rf.fit(X_train_res, y_train_res)

# # Make predictions
# y_pred_rf = rf.predict(X_test)

# # Evaluate the RandomForest model
# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
# print(classification_report(y_test, y_pred_rf))

# # Confusion matrix
# conf_matrix = confusion_matrix(y_test, y_pred_rf)
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=df['Deficiency'].astype('category').cat.categories, yticklabels=df['Deficiency'].astype('category').cat.categories)
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.show()



import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # Import pickle to save the model

# Create the DataFrame
data = {
    'Hemoglobin': [13.5, 15.2, 10.0, 12.5, 16.0, 9.0, 13.8, 11.5, 14.0, 12.0, 13.0, 11.0, 12.8, 14.5, 9.8, 12.7, 13.2, 15.0, 10.5, 14.2, 12.2, 11.8, 13.6, 15.5, 10.2, 11.6],
    'RBC': [4.7, 5.1, 3.8, 4.2, 5.6, 3.2, 4.8, 4.0, 4.9, 3.9, 4.5, 3.6, 4.3, 5.2, 3.4, 4.1, 4.7, 5.3, 3.9, 5.0, 4.1, 3.7, 4.6, 5.4, 3.5, 4.0],
    'WBC': [6.0, 5.5, 8.0, 7.0, 4.5, 10.0, 5.0, 9.5, 5.8, 8.5, 6.2, 9.2, 7.5, 5.0, 9.8, 6.8, 6.0, 4.8, 8.8, 5.5, 7.2, 8.0, 6.3, 4.6, 9.0, 9.2],
    'Platelets': [250, 280, 180, 270, 320, 150, 290, 220, 275, 200, 260, 180, 240, 310, 170, 230, 290, 300, 190, 295, 240, 220, 270, 310, 170, 210],
    'Ferritin': [120, 200, 30, 50, 300, 10, 100, 40, 150, 35, 60, 25, 45, 180, 12, 55, 95, 220, 20, 140, 50, 35, 110, 250, 18, 40],
    'Cholesterol': [180, 190, 220, 240, 150, 260, 200, 210, 175, 230, 190, 250, 210, 160, 270, 200, 180, 170, 240, 180, 210, 220, 190, 160, 260, 215],
    'Glucose': [90, 100, 160, 140, 110, 180, 95, 150, 100, 170, 115, 160, 140, 90, 190, 145, 100, 105, 175, 90, 135, 155, 100, 95, 185, 150],
    'Deficiency': ['Iron Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'None', 'Iron Deficiency', 'None', 'Vitamin B12 Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'Iron Deficiency', 'Vitamin B12 Deficiency', 'None', 'Iron Deficiency', 'Vitamin D Deficiency', 'None', 'None', 'Iron Deficiency', 'None', 'Vitamin D Deficiency', 'Iron Deficiency', 'None', 'None', 'Iron Deficiency', 'Vitamin B12 Deficiency']
}

df = pd.DataFrame(data)

# Convert categorical 'Deficiency' column to numeric labels
df['Deficiency'] = df['Deficiency'].astype('category').cat.codes

# Define features (X) and target (y)
X = df.drop(columns=['Deficiency'])
y = df['Deficiency']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use SMOTE to handle class imbalance
smote = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale the features
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_res, y_train_res)

# Save the trained model
model_path = 'random_forest_blood_report_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf, file)

# Load the model (optional, to check if it loads correctly)
# with open(model_path, 'rb') as file:
#     loaded_rf = pickle.load(file)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the RandomForest model
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=df['Deficiency'].astype('category').cat.categories, 
            yticklabels=df['Deficiency'].astype('category').cat.categories)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Random Forest Model')
plt.show()
