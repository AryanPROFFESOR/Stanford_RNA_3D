import torch
import torch.nn as nn

class SequenceEncoder(nn.Module):
    def __init__(self, vocab=4, dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=8, batch_first=True),
            num_layers=4,
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.encoder(x)


class PairwiseModule(nn.Module):
    def __init__(self, dim=256, pair_dim=128):
        super().__init__()
        self.linear = nn.Linear(dim * 2, pair_dim)

    def forward(self, h):
        B, L, D = h.shape
        hi = h.unsqueeze(2).expand(B, L, L, D)
        hj = h.unsqueeze(1).expand(B, L, L, D)
        pair = torch.cat([hi, hj], dim=-1)
        return self.linear(pair)


class StructureModule(nn.Module):
    def __init__(self, pair_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, pair_repr):
        coords = self.mlp(pair_repr).mean(dim=2)
        return coords


class RNAFoldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = SequenceEncoder()
        self.pair_module = PairwiseModule()
        self.structure = StructureModule()

    def forward(self, x):
        h = self.seq_encoder(x)
        pair = self.pair_module(h)
        coords = self.structure(pair)
        return coords
