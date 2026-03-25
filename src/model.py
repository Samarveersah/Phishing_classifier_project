import torch
from torch import nn


class HybridPhishingCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int,
        embedding_dim: int = 64,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size=size) for size in kernel_sizes]
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        combined_dim = len(kernel_sizes) * num_filters + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        conv_outputs = [torch.relu(conv(embedded)) for conv in self.convs]
        pooled = [torch.max(output, dim=2).values for output in conv_outputs]
        url_representation = torch.cat(pooled, dim=1)
        projected_features = self.feature_projection(features)
        combined = torch.cat([url_representation, projected_features], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits
