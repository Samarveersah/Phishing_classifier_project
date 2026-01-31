import pandas as pd
from sklearn.model_selection import train_test_split

from model import PhishingModel


# Load dataset
data = pd.read_parquet(r"C:\Users\samar\OneDrive\Desktop\phishing_url_detector\Training.parquet")


# Separate features and target
X = data.drop(columns=["status", "url"])
y = data["status"].map({"legitimate": 0, "phishing": 1})


# Convert to numeric
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize model
model = PhishingModel()

# Train all models and select best
model.train_all(X_train, y_train, X_test, y_test)

# Save best model
model.save_model()
