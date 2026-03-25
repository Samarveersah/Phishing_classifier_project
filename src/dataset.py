from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.feature_extraction import extract_url_features
from src.preprocessing import build_attention_mask, encode_url


class URLDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocab: Dict[str, int], max_length: int):
        self.frame = frame.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        url = str(row["url"])
        tokens = encode_url(url, self.vocab, self.max_length)
        features = extract_url_features(url)
        label = int(row["label"])
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "mask": torch.tensor(build_attention_mask(tokens), dtype=torch.float32),
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def infer_feature_dim() -> int:
    return int(np.asarray(extract_url_features("https://example.com")).shape[0])
