from dataclasses import dataclass

import torch

from src.config import CONFIG_PATH, MAX_URL_LENGTH, MODEL_PATH, VOCAB_PATH
from src.feature_extraction import FEATURE_NAMES, extract_url_features
from src.model import HybridPhishingCNN
from src.preprocessing import encode_url
from src.utils import load_json


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probability: float
    risk_level: str
    handcrafted_features: dict[str, float]


class PredictionService:
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH, config_path=CONFIG_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = load_json(config_path)
        self.vocab = load_json(vocab_path)
        self.max_length = int(self.config.get("max_url_length", MAX_URL_LENGTH))
        self.model = HybridPhishingCNN(
            vocab_size=len(self.vocab),
            feature_dim=int(self.config["feature_dim"]),
            embedding_dim=int(self.config["embedding_dim"]),
            num_filters=int(self.config["num_filters"]),
            kernel_sizes=tuple(self.config["kernel_sizes"]),
            hidden_dim=int(self.config["hidden_dim"]),
            dropout=float(self.config["dropout"]),
        )
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, url: str) -> PredictionResult:
        tokens = torch.tensor([encode_url(url, self.vocab, self.max_length)], dtype=torch.long).to(self.device)
        features = torch.tensor([extract_url_features(url)], dtype=torch.float32).to(self.device)
        probability = float(torch.sigmoid(self.model(tokens, features)).item())
        label = "Phishing" if probability >= 0.5 else "Legitimate"
        confidence = probability if probability >= 0.5 else 1 - probability
        risk_level = _to_risk_level(probability)
        feature_values = extract_url_features(url)
        feature_map = {name: float(value) for name, value in zip(FEATURE_NAMES, feature_values)}
        return PredictionResult(
            label=label,
            confidence=confidence,
            probability=probability,
            risk_level=risk_level,
            handcrafted_features=feature_map,
        )


def _to_risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    if probability >= 0.5:
        return "Medium"
    return "Low"
