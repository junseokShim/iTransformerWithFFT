import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, T, D]
        x_fft = torch.fft.rfft(x, dim=1)                 # [B, T//2+1, D]
        x_ifft = torch.fft.irfft(x_fft, n=x.size(1), dim=1)  # [B, T, D]
        return self.proj(x_ifft)


class FourierBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.attn = FourierAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Residual + Norm + Attention
        x = x + self.ff(self.norm2(x))    # Residual + Norm + FFN
        return x


class FourierTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=64, depth=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.layers = nn.Sequential(*[FourierBlock(d_model, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T, input_dim]
        x = self.embedding(x)  # [B, T, d_model]
        x = self.layers(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global Average Pooling
        return self.classifier(x)
