import os
import random
import numpy as np
import math
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from intellino.core.neuron_cell import NeuronCells
from PIL import Image
from tqdm import tqdm

# CLAHE 적용 Transform 클래스
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        np_img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        np_img = clahe.apply(np_img)
        return Image.fromarray(np_img)

# Feature Extractor 정의
class FingerVeinFeatureExtractor(nn.Module):
    def __init__(self):
        super(FingerVeinFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 153, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(153, 512, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 768, kernel_size=5, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(768, 1024, kernel_size=(4,15), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x

# Feature Classifier 정의
class FeatureClassifier(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(FeatureClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# FV-USM Dataset 로더
class FvUsmDataset(Dataset):
    def __init__(self, folder_list, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        for label_idx, folder in enumerate(folder_list):
            img_names = os.listdir(folder)
            for img_name in img_names:
                if img_name.lower().endswith('.jpg'):
                    img_path = os.path.join(folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 흑백 변환
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transform 설정
data_transform = transforms.Compose([
    CLAHETransform(clip_limit=2.0, tile_grid_size=(8,8)),
    transforms.Resize((65, 153)),
    transforms.ToTensor()
])

# FV-USM 데이터셋 폴더 경로
root_dir = r"dataset/Published_database_FV-USM_Dec2013/1st_session/extractedvein"

# 폴더 리스트 가져오기
all_folders = [os.path.join(root_dir, d) for d in sorted(os.listdir(root_dir)) if os.path.isdir(os.path.join(root_dir, d))]

# 랜덤 셔플 후 90% train, 10% test 분리
random.seed(42)  # 재현성 확보
random.shuffle(all_folders)
split_idx = int(0.9 * len(all_folders))
train_folders = all_folders[:split_idx]
test_folders = all_folders[split_idx:]

# 데이터셋 생성
train_dataset = FvUsmDataset(folder_list=train_folders, transform=data_transform)
test_dataset = FvUsmDataset(folder_list=test_folders, transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Feature Extractor 및 Classifier 초기화
feature_extractor = FingerVeinFeatureExtractor()
model = FeatureClassifier(feature_extractor, num_classes=len(train_folders))

# Optimizer, Loss
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# CNN 학습 (Early Stopping 적용)
print("Start CNN Training...")
early_stopping_patience = 3
best_loss = float('inf')
patience_counter = 0

for epoch in range(100):  # 최대 50 에폭
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

# Feature Extractor만 사용
feature_extractor = model.feature_extractor
# Feature Extractor 저장
save_path = './finger_vein_feature_extractor.pth'
torch.save(feature_extractor.state_dict(), save_path)
print(f"Feature extractor saved at {save_path}")

# 저장된 Feature Extractor 로딩
feature_extractor = FingerVeinFeatureExtractor()
feature_extractor.load_state_dict(torch.load('./finger_vein_feature_extractor.pth'))
feature_extractor.eval()


# NeuronCells 초기화
number_of_neuron_cells = 738
length_of_input_vector = 1024
neuron_cells = NeuronCells(number_of_neuron_cells=number_of_neuron_cells,
                           length_of_input_vector=length_of_input_vector,
                           measure="manhattan")

# KNN 학습
print("Start NeuronCells Training...")
for idx, (img, label) in enumerate(tqdm(train_loader, desc="Training")):
    with torch.no_grad():
        feature = feature_extractor(img)
        feature_vector = feature.squeeze().cpu().numpy()
    for fvec, lbl in zip(feature_vector, label):
        is_finish = neuron_cells.train(vector=fvec, target=lbl.item())
        if is_finish:
            print("NeuronCells training completed early.")
            break

# Test
print("Start Testing...")
cnt = 0
correct = 0
for idx, (img, label) in enumerate(tqdm(test_loader, desc="Testing")):
    if idx == 1000:
        accuracy = correct / cnt * 100
        print(f"Accuracy: {accuracy:.2f}%")
        break
    cnt += 1
    with torch.no_grad():
        feature = feature_extractor(img)
        feature_vector = feature.squeeze().cpu().numpy()
    predict_label = neuron_cells.inference(vector=feature_vector)
    print(f"Test sample {idx}: Label={label.item()}, Predicted={predict_label}")
    if predict_label == label.item():
        correct += 1
