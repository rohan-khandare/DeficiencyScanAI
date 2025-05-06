import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# Load the dataset
data = pd.read_csv('D:/IPCV-cp/data/nutritional_deficiency_dataset2.csv')

# Handling missing values (if any)
data.fillna(method='ffill', inplace=True)

# Categorical and numerical columns
categorical_columns = [
    'Fatigue', 'Headaches', 'Cramps', 'Veg_Fruit_Intake', 'Skin_Problems', 
    'Sleep_Hours', 'Joint_Pain', 'Hair_Loss', 'Mood_Swings', 
    'Digestive_Issues', 'Exercise_Frequency', 'Sun_Exposure', 'Thirst_Frequency'
]
target_column = 'Deficiency_Present'

# Preprocessing: Encoding categorical variables and scaling numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ]
)

# Model Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),  # Scaling to normalize data distribution
    ('classifier', RandomForestClassifier(random_state=42))
])

# Features and target
X = data[categorical_columns]
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Cross-validation and Grid Search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Save the best model
with open('best_random_forest_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
try:
    roc_score = roc_auc_score(y_test, best_model.predict_proba(X_test), multi_class='ovr')
    print(f"AUC-ROC Score: {roc_score}")
except ValueError:
    print("AUC-ROC Score: Cannot compute due to binary class output only.")

# Load the saved model (optional step to verify)
with open('best_random_forest_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
    loaded_y_pred = loaded_model.predict(X_test)
    print("Loaded Model Classification Report:\n", classification_report(y_test, loaded_y_pred))
