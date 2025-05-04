import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 1. Fourier Self-Attention
class FourierSelfAttention(nn.Module):
    def __init__(self, d_model, heads=32, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (d_model // heads) ** -0.5
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

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

# ✅ 2. Multi-Scale Patch Embedding
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, patch_sizes=[4, 8, 16]):
        super().__init__()
        self.patch_embeddings = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=ps, stride=ps, padding=ps//2)
            for ps in patch_sizes
        ])


    def forward(self, x):
        x = x.transpose(1, 2)  # [B, T, D] → [B, D, T]
        patches = [pe(x) for pe in self.patch_embeddings]  # [B, H, T']
        out = torch.cat(patches, dim=2)  # 시간축 결합
        return out.transpose(1, 2)       # [B, T', H]

# ✅ 3. Mamba-like Feedforward Block
class MambaBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))  # Residual

# ✅ 4. Cross-Attention (내생/외생 변수 융합)
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

    def forward(self, x, z):
        B, T, D = x.shape
        Q = self.q_proj(x).view(B, T, self.heads, D // self.heads).transpose(1, 2)
        K, V = self.kv_proj(z).chunk(2, dim=-1)
        K = K.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        V = V.view(B, T, self.heads, D // self.heads).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        out = (attn_weights @ V).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(self.dropout(out))

# ✅ 5. 최종 통합 모델
class AdvancedITransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_heads=8, patch_sizes=[4, 8, 16]):
        super().__init__()
        self.patch_embed = MultiScalePatchEmbedding(input_dim, hidden_dim, patch_sizes)
        self.fourier_attn = FourierSelfAttention(hidden_dim, heads=num_heads)
        self.mamba = MambaBlock(hidden_dim)
        self.cross_attn = CrossAttentionBlock(hidden_dim, heads=num_heads)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, z=None):
        # x: [B, T, D], z(optional): [B, T, D]
        x = self.patch_embed(x)         # → [B, T', H]
        x = self.fourier_attn(x)
        x = self.mamba(x)
        if z is not None:
            z = self.patch_embed(z)
            x = self.cross_attn(x, z)
        x = x.mean(dim=1)
        return self.classifier(x)
