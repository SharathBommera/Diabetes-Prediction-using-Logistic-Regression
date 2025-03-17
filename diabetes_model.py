import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
file_path = 'D:/projects/Diabetes_Prediction/Dataset of Diabetes .csv'
diabetes_data = pd.read_csv(file_path)4

# Drop irrelevant columns
diabetes_data_cleaned = diabetes_data.drop(columns=["ID", "No_Pation"])

# Encode categorical variables
label_encoder = LabelEncoder()
diabetes_data_cleaned["Gender"] = label_encoder.fit_transform(diabetes_data_cleaned["Gender"])
diabetes_data_cleaned["CLASS"] = label_encoder.fit_transform(diabetes_data_cleaned["CLASS"])

# Separate features and target
X = diabetes_data_cleaned.drop(columns=["CLASS"])
y = diabetes_data_cleaned["CLASS"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a logistic regression model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluate the model
y_pred = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and scaler
joblib.dump(logistic_model, "diabetes_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")
