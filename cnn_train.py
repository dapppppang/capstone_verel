import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Feature Extractor 및 Classifier 초기화
model = FingerVeinFeatureExtractor()

# Optimizer, Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# CNN 학습 (Early Stopping 적용)
print("Start CNN Training...")
early_stopping_patience = 3
best_loss = float('inf')
patience_counter = 0

for epoch in range(100):  # 최대 100 에폭
    running_loss = 0.0
    model.train()
    for img, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print("Early stopping triggered!")
        break

# Feature Extractor 저장
save_path = 'finger_vein_feature_extractor.pth'
torch.save(model.state_dict(), save_path)
print(f"Feature extractor saved at {save_path}")
