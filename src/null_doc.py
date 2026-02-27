import torch
import torch.nn as nn
import torch.nn.functional as F


class NullDocumentEmbedding(nn.Module):

    def __init__(self, dim: int = 768):
        super().__init__()
        self.e_null = nn.Parameter(torch.randn(dim) * 0.01)

    def forward(self, query_vecs: torch.Tensor):
        e_norm = F.normalize(self.e_null.unsqueeze(0), p=2, dim=-1)
        return (query_vecs @ e_norm.T).squeeze(-1)

    def embedding_norm(self):
        return self.e_null.detach().norm().item()
