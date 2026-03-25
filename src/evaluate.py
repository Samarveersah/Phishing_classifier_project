from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


@torch.no_grad()
def evaluate_model(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    probabilities = []
    predictions = []
    labels = []

    for batch in dataloader:
        tokens = batch["tokens"].to(device)
        features = batch["features"].to(device)
        batch_labels = batch["label"].to(device)

        logits = model(tokens, features)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        probabilities.extend(probs.cpu().numpy().tolist())
        predictions.extend(preds.cpu().numpy().tolist())
        labels.extend(batch_labels.cpu().numpy().tolist())

    labels_np = np.array(labels)
    predictions_np = np.array(predictions)
    probabilities_np = np.array(probabilities)

    metrics = {
        "accuracy": float(accuracy_score(labels_np, predictions_np)),
        "precision": float(precision_score(labels_np, predictions_np, zero_division=0)),
        "recall": float(recall_score(labels_np, predictions_np, zero_division=0)),
        "f1": float(f1_score(labels_np, predictions_np, zero_division=0)),
    }
    if len(np.unique(labels_np)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels_np, probabilities_np))
    else:
        metrics["roc_auc"] = 0.0
    return metrics
