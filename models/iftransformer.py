import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Fourier Self-Attention (실수 연산 + Head 지원)
class FourierSelfAttention(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (d_model // heads) ** -0.5

        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, C // self.heads).transpose(1, 2), qkv)

        q_fft = torch.fft.rfft(q.float(), dim=-2)
        k_fft = torch.fft.rfft(k.float(), dim=-2)

        attn = (q_fft.conj() * k_fft).sum(dim=-1).real * self.scale
        attn = F.softmax(attn, dim=-1)

        v_fft = torch.fft.rfft(v.float(), dim=-2)
        out_fft = attn.unsqueeze(-1) * v_fft
        out = torch.fft.irfft(out_fft, n=T, dim=-2)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.to_out(out)

# 2. PreNorm Fourier Encoder Layer
class FourierEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=32, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FourierSelfAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # PreNorm + Residual
        x = x + self.ff(self.norm2(x))    # PreNorm + Residual
        return x

# 3. Learnable Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=14):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# 4. 최종 Transformer Classifier
class IFTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_heads = 32,  n_layers=6, max_len=14, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = PositionalEncoding(hidden_dim, max_len)

        self.layers = nn.ModuleList([
            FourierEncoderLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x)  # [B, 1, D] -> [B, 1, H]
        #x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)  # Global Average Pooling
        return self.fc_out(x)
