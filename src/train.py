import argparse
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import (
    CONFIG_PATH,
    MAX_URL_LENGTH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    VOCAB_PATH,
)
from src.data_loader import load_dataset, save_splits, train_val_test_split
from src.dataset import URLDataset, infer_feature_dim
from src.evaluate import evaluate_model
from src.model import HybridPhishingCNN
from src.preprocessing import build_char_vocab
from src.utils import ensure_dir, save_json, set_seed


@dataclass
class TrainingConfig:
    data_path: Optional[str] = None
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_url_length: int = MAX_URL_LENGTH
    embedding_dim: int = 64
    num_filters: int = 64
    hidden_dim: int = 64
    dropout: float = 0.3
    min_char_frequency: int = 1


def train(config: TrainingConfig) -> dict:
    set_seed(RANDOM_STATE)
    frame = load_dataset(config.data_path)
    train_frame, val_frame, test_frame = train_val_test_split(frame)
    save_splits(train_frame, val_frame, test_frame)

    vocab = build_char_vocab(train_frame["url"].tolist(), min_frequency=config.min_char_frequency)
    feature_dim = infer_feature_dim()

    train_dataset = URLDataset(train_frame, vocab, config.max_url_length)
    val_dataset = URLDataset(val_frame, vocab, config.max_url_length)
    test_dataset = URLDataset(test_frame, vocab, config.max_url_length)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridPhishingCNN(
        vocab_size=len(vocab),
        feature_dim=feature_dim,
        embedding_dim=config.embedding_dim,
        num_filters=config.num_filters,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    positive_count = float(train_frame["label"].sum())
    negative_count = float(len(train_frame) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_f1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(tokens, features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tokens.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_metrics = evaluate_model(model, val_loader, device)
        val_metrics["loss"] = float(train_loss)
        val_metrics["epoch"] = epoch
        history.append(val_metrics)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    ensure_dir(MODEL_PATH.parent)
    torch.save(best_state, MODEL_PATH)
    save_json(VOCAB_PATH, vocab)

    artifact_config = asdict(config)
    artifact_config.update(
        {
            "feature_dim": feature_dim,
            "kernel_sizes": [3, 4, 5],
            "random_state": RANDOM_STATE,
        }
    )
    save_json(CONFIG_PATH, artifact_config)

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(model, test_loader, device)
    metrics_payload = {
        "validation_history": history,
        "best_validation_f1": best_val_f1,
        "test_metrics": test_metrics,
    }
    save_json(METRICS_PATH, metrics_payload)
    return metrics_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the hybrid CNN phishing URL detector.")
    parser.add_argument("--data-path", type=str, default=None, help="Optional path to CSV or parquet dataset.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-url-length", type=int, default=MAX_URL_LENGTH)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = TrainingConfig(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_url_length=args.max_url_length,
    )
    metrics = train(config)
    print("Training complete.")
    print(metrics)


if __name__ == "__main__":
    main()
