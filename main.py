import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import PhishingModel


# Load dataset
data = pd.read_parquet("Training.parquet")

# Separate features and target
X = data.drop(columns=["status", "url"])
y = data["status"]

# Convert all feature columns to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
model = PhishingModel()

# Train model
model.train(X_train, y_train)

# Predict
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save trained model
model.save_model()
