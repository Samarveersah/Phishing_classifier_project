import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class PhishingModel:

    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                use_label_encoder=False
            )
        }

        self.best_model = None
        self.best_accuracy = 0
        self.best_model_name = None

    def train_all(self, X_train, y_train, X_test, y_test):

        print("\nTraining models...\n")

        for name, model in self.models.items():

            print(f"Training {name}...")

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            print(f"{name} Accuracy: {accuracy:.4f}\n")

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name

        print("===================================")
        print("Best Model Selected:", self.best_model_name)
        print("Best Accuracy:", round(self.best_accuracy, 4))
        print("===================================\n")

    def save_model(self, file_name="phishing_model.pkl"):

        if self.best_model is None:
            print("No model trained yet. Train model first.")
            return

        joblib.dump(self.best_model, file_name)

        print(f"Model saved successfully as {file_name}")
