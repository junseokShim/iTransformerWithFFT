import torch

# 예측 함수 정의
def predict(model, X_tensor, target):    
    model.eval()
    with torch.no_grad():
        # Load the best weights
        model.load_state_dict(torch.load(f'./weights/{target}_best_model_weights.pt'))
        
        outputs = model(X_tensor)
        return outputs.argmax(dim=1).numpy()