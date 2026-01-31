import joblib
from sklearn.ensemble import RandomForestClassifier


class PhishingModel:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, path="phishing_model.pkl"):
        joblib.dump(self.model, path)

    def load_model(self, path="phishing_model.pkl"):
        self.model = joblib.load(path)
