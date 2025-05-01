import torch

# 예측 함수 정의
def predict(model, X_tensor):
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        return outputs.argmax(dim=1).numpy()