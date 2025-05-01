import torch
from torch.utils.data import random_split, DataLoader

def train_model(model, dataloader, criterion, optimizer, col, epochs=20):
    # 내부에서 validation set 분리
    dataset = dataloader.dataset
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            #print(f"Epoch: {epoch+1}, Batch Loss: {round(loss.item(), 4)}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch: {epoch+1}, Avg Val Loss: {round(avg_val_loss, 4)}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f'weights/{col}_best_model_weights.pt')
            print(f"✅ Best model saved for {col}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= 15:
            print("Early stopping triggered.")
            return

    print("Training complete.")
