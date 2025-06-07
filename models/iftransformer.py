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
    

class MultiScaleFourier(nn.Module):
    def __init__(self, hidden_dim, scales=[1,2,4]):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=scale, padding='same')
            for scale in scales
        ])

    def forward(self, x):
        fft_feats = torch.fft.rfft(x, dim=1).real
        multi_feats = [conv(fft_feats.transpose(1,2)).transpose(1,2) for conv in self.convs]
        return torch.cat(multi_feats, dim=-1)


class PatchEmbedding(nn.Module):
    def __init__(self, hidden_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(hidden_dim * patch_size, hidden_dim)

    def forward(self, x):
        B, T, D = x.shape
        assert T % self.patch_size == 0
        x = x.reshape(B, T // self.patch_size, self.patch_size * D)
        return self.linear(x)


class SimpleSSM(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x_ssm = self.conv1d(x.transpose(1,2)).transpose(1,2)
        return self.norm(x + x_ssm)


class ShapeAwareEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.shape_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        shape_feat = torch.diff(x, dim=1, prepend=x[:, :1, :])
        return x + self.shape_linear(shape_feat)



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


class AdvancedIFTransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_heads=8, n_layers=4, patch_size=2, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = PositionalEncoding(hidden_dim, max_len=100)

        self.patch_embed = PatchEmbedding(hidden_dim, patch_size)
        self.shape_embed = ShapeAwareEmbedding(hidden_dim)
        self.multiscale_fourier = MultiScaleFourier(hidden_dim)
        
        encoder_dim = hidden_dim * (len(self.multiscale_fourier.scales))
        
        self.layers = nn.ModuleList([
            FourierEncoderLayer(encoder_dim, num_heads, dropout) for _ in range(n_layers)
        ])

        self.ssm = SimpleSSM(encoder_dim)

        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.pos_embed(x)
        x = self.shape_embed(x)
        x = self.patch_embed(x)
        x = self.multiscale_fourier(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ssm(x)

        x = x.mean(dim=1)
        return self.fc_out(x)


class IFTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_heads=64, n_layers=6, max_len=14, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = PositionalEncoding(hidden_dim, max_len)
        self.patch_embed = PatchEmbedding(hidden_dim, patch_size = 1) # 0.587
        self.shape_embed = ShapeAwareEmbedding(hidden_dim)

        self.layers = nn.ModuleList([
            FourierEncoderLayer(hidden_dim, num_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.ssm = SimpleSSM(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # self.fc_out = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim//2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim//2, num_classes)
        # )
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x)  # [B, 1, D] -> [B, 1, H]
        #x = self.pos_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=1)  # Global Average Pooling
        return self.fc_out(x)
