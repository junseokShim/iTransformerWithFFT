import torch
import torch.nn as nn
import torch.nn.functional as F

# Fourier Self-Attention (실수 연산 기반)
class FourierSelfAttention(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.3):
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
        attn = F.softmax(attn + 1e-6, dim=-1)

        v_fft = torch.fft.rfft(v.float(), dim=-2)
        out_fft = attn.unsqueeze(-1) * v_fft
        out = torch.fft.irfft(out_fft, n=T, dim=-2)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.to_out(out)


# ▶ Transformer Block with FourierSelfAttention + FFN + Residual + LayerNorm
class FourierTransformerBlock(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.3):
        super().__init__()
        self.attn = FourierSelfAttention(d_model, heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))  # Residual + Norm
        x = self.norm2(x + self.ffn(x))   # Residual + Norm
        return x


# ▶ Classifier (Encoder + Mean Pooling + Classifier Head)
class FFTformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_heads, num_layers=12, pretrained_encoder = None):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        if pretrained_encoder:
            self.encoder_layers = pretrained_encoder
        else:  
            self.encoder_layers = nn.Sequential(*[
                FourierTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
            ])
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.encoder_layers(x)
        x = x.mean(dim=1)  # global average pooling
        return self.fc_out(x)


# ▶ Pretrainer (Encoder + Decoder with MLP Reconstruction)
class FFTformerPretrainer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers=6):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.Sequential(*[
            FourierTransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 복원용
        )

    def forward(self, x, mask):
        """
        x: [B, T, D], mask: [B, T]
        """
        x = self.fc_in(x)
        x = self.encoder_layers(x)
        x = self.decoder(x)
        return x  # 전체 출력, 손실 계산은 마스크 위치만 사용
