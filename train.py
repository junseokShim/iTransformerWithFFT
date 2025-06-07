import torch
from torch.utils.data import random_split, DataLoader
from sklearn.metrics import f1_score
import torch.nn as nn

def train_model(model, dataloader, criterion, optimizer, col, epochs=20):
    # 내부에서 validation set 분리
    dataset = dataloader.dataset
    val_size = int(len(dataset) * 0.4)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=False)

    best_val_loss = float('inf')
    best_val_f1 = 0
    early_stop_counter = 0

    base_lr = optimizer.param_groups[0]['lr']
    warmup_epochs = 10  # 고정된 warm-up epoch 수

    for epoch in range(epochs):
        # Warm-up: 선형 증가
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]['lr'] = warmup_lr
            print(f"Epoch {epoch+1}: Warm-up LR = {warmup_lr:.6f}")
        else:
            optimizer.param_groups[0]['lr'] = base_lr

        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_losses = []
        val_f1_scores = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                val_outputs = model(x_val)
                val_loss = criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())
                val_predictions = torch.argmax(val_outputs, dim=1)
                val_f1 = f1_score(y_val, val_predictions, average='weighted')
                val_f1_scores.append(val_f1)
            val_f1_scores.append(val_f1)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_f1 = sum(val_f1_scores) / len(val_f1_scores)
        print(f"Epoch: {epoch+1}, Avg Val Loss: {round(avg_val_loss, 4)}, Avg Val F1 Score: {round(avg_val_f1, 4)}")

        # Early stopping
        if avg_val_f1 > best_val_f1:
            best_val_f1 = avg_val_f1
            early_stop_counter = 0
            torch.save(model.state_dict(), f'weights/{col}_best_model_weights.pt')
            print(f"✅ Best model saved for {col}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= 15:
            print("Early stopping triggered.")
            return best_val_f1

    print("Training complete.")
    return best_val_loss


def train_pretraining_model_with_val(model, dataloader, optimizer, epochs=100, mask_ratio=0.15, patience=100, save_path="pretrained_itransformer.pt"):
    criterion = nn.MSELoss()

    # ✅ Train/Validation Split
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
        total_loss = 0
        for x_batch, in train_loader:
            optimizer.zero_grad()
            x_batch = x_batch.to(torch.float32)
            x_masked, mask = random_mask_input(x_batch, mask_ratio)
            x_recon = model(x_masked, mask)
            loss = criterion(x_recon[mask], x_batch[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ✅ Validation Phase
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for x_val, in val_loader:
                x_val = x_val.to(torch.float32)
                x_masked, mask = random_mask_input(x_val, mask_ratio)

                if mask.sum() == 0:
                    continue  # 마스크된 위치가 없으면 손실 계산 skip

                x_recon = model(x_masked, mask)
                val_loss = criterion(x_recon[mask], x_val[mask])
                val_loss_total += val_loss.item()
        avg_val_loss = val_loss_total / len(val_loader)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ✅ Early Stopping & Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (val loss: {avg_val_loss:.4f})")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"⛔ Early stopping triggered at epoch {epoch+1}")
            break

    print("🎯 Pretraining complete.")


def random_mask_input(x, mask_ratio=0.1):
    """
    입력 시계열 x [B, T, D] 중 일부를 mask (0으로) 처리하고, mask 위치를 반환
    """
    B, T, D = x.shape
    mask = torch.rand(B, T) < mask_ratio  # [B, T]
    x_masked = x.clone()
    x_masked[mask.unsqueeze(-1).expand(-1, -1, D)] = 0.0
    return x_masked, mask
