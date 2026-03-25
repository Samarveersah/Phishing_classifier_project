from typing import Dict, List

import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def normalize_url(url: str) -> str:
    return (url or "").strip().lower()


def build_char_vocab(urls: List[str], min_frequency: int = 1) -> Dict[str, int]:
    counts = {}
    for url in urls:
        for character in normalize_url(url):
            counts[character] = counts.get(character, 0) + 1

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for character, count in sorted(counts.items(), key=lambda item: item[0]):
        if count >= min_frequency:
            vocab[character] = len(vocab)
    return vocab


def encode_url(url: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    normalized = normalize_url(url)
    encoded = [vocab.get(character, vocab[UNK_TOKEN]) for character in normalized[:max_length]]
    if len(encoded) < max_length:
        encoded.extend([vocab[PAD_TOKEN]] * (max_length - len(encoded)))
    return encoded


def build_attention_mask(tokens: List[int], pad_index: int = 0) -> List[int]:
    return [0 if token == pad_index else 1 for token in tokens]


def encode_url_tensor(url: str, vocab: Dict[str, int], max_length: int) -> torch.Tensor:
    return torch.tensor(encode_url(url, vocab, max_length), dtype=torch.long)
