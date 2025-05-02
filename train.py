import torch

# 학습 함수 정의
def train_model(model, dataloader, criterion, optimizer, col, epochs=20):
    model.train()
    best_loss = float('inf')
    early_stop_counter = 0
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            # Print training loss for each batch
            print(f"Epoch: {epoch+1}, Batch Loss: {loss.item()}")
            
            # Check if the current loss is the best so far
            if loss < best_loss:
                best_loss = loss
                early_stop_counter = 0
                # Save the model weights
                torch.save(model.state_dict(), f'weights/{col}_best_model_weights.pt')
            else:
                early_stop_counter += 1
            
            # Check if early stopping condition is met
            if early_stop_counter >= 20:
                print("Early stopping triggered.")
                return
            
    print("Training complete.")
            