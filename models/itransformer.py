import torch
import torch.nn as nn
import torch.nn.functional as F

# Fourier Self-Attention 정의 (실수 연산만 사용)
class FourierSelfAttention(nn.Module):
    def __init__(self, d_model, heads=32, dropout=0.1):
        super().__init__()
        self.heads = heads
        if d_model <= 0:
            d_model = 1  # Set d_model to a small positive value
        self.scale = nn.Parameter(torch.tensor((d_model // self.heads) ** -0.5))

        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, C // self.heads).transpose(1, 2), qkv)

        # Fourier Transform on queries and keys
        q_fft = torch.fft.rfft(q.float(), dim=-2)
        k_fft = torch.fft.rfft(k.float(), dim=-2)

        attn = (q_fft.conj() * k_fft).sum(dim=-1).real * self.scale
        attn = F.softmax(attn, dim=-1)

        v_fft = torch.fft.rfft(v.float(), dim=-2)
        out_fft = attn.unsqueeze(-1) * v_fft
        out = torch.fft.irfft(out_fft, n=T, dim=-2)

        out = out.transpose(1, 2).reshape(B, T, C)
        return self.to_out(out)
            

    
# Transformer 모델 정의 (FourierSelfAttention + iTransformer 입력 방식 적용)
class ITransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim, num_heads, pretrained_encoder=None):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fourier_attn = pretrained_encoder if pretrained_encoder else FourierSelfAttention(hidden_dim, num_heads)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x.squeeze(1)).unsqueeze(1)
        x = self.fourier_attn(x)
        x = x.mean(dim=1)  # 평균 풀링
        return self.fc_out(x)
    

# Transformer Pretrainer 정의 (FourierSelfAttention + iTransformer 입력 방식 적용)
class ITransformerPretrainer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.encoder = FourierSelfAttention(hidden_dim, num_heads)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 복원: 원래 차원
        )

    def forward(self, x, mask):
        """
        x: [B, T, D], mask: [B, T] → 복원 loss는 masked 위치에 대해서만 계산
        """
        x_encoded = self.fc_in(x.squeeze(1)).unsqueeze(1)
        x_encoded = self.encoder(x_encoded)
        x_decoded = self.decoder(x_encoded)  # [B, T, D]

        return x_decoded  # 전체 출력 (손실 계산 시 mask로 걸러냄)

