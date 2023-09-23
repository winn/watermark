import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# Load the dataset (replace the path with the actual path to your dataset)
# If your dataset is in a different format (e.g., Excel), adjust the read method accordingly
data = pd.read_excel("Survey_KKU (Responses).xlsx")

# Preprocess the data
# 1. Handle missing values (filling with mean in this example)
data["เสต็ก Steak"].fillna(data["เสต็ก Steak"].mean(), inplace=True)

# 2. Drop unnecessary columns and separate features and target
X = data.drop(columns=['ภูเขา Mountain', 'nickname', 'Timestamp'])
y = data['ภูเขา Mountain']

# 3. Binarize the target variable
y_binarized = (y > 3).astype(int)

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarized, test_size=0.2, random_state=42)

# Train the Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)

# Save the trained model and scaler to .joblib files
# Save the model and scaler
with open('gb_model.pkl', 'wb') as model_file:
    pickle.dump(gb_clf, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)


# joblib.dump(gb_clf, 'gb_model.joblib')
# joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler have been saved successfully!")
