
from sklearn.preprocessing import StandardScaler

# Fourier Attention 정의
import torch
import torch.nn as nn

# Fourier Self-Attention 정의 (실수 연산만 사용)
class FourierSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        Q = self.proj_q(x)  # [B, T, D]
        K = self.proj_k(x)
        V = self.proj_v(x)

        # Fourier 변환 (실수부만 사용)
        Q_fft = torch.fft.rfft(Q.float(), dim=1).real
        K_fft = torch.fft.rfft(K.float(), dim=1).real
        V_fft = torch.fft.rfft(V.float(), dim=1).real

        attn_scores = torch.matmul(Q_fft, K_fft.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)

        output_fft = torch.matmul(attn_weights, V_fft)
        output = torch.fft.irfft(output_fft.to(torch.complex64), n=x.size(1), dim=1)

        return self.out(output)

# Transformer 모델 정의 (FourierSelfAttention 적용)
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fourier_attn = FourierSelfAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc_in(x).unsqueeze(1)  # [batch, seq_len=1, features]
        x = self.fourier_attn(x)
        x = x.mean(dim=1)  # 평균 풀링
        return self.fc_out(x)

