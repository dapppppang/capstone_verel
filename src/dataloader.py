import os
import random
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


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
root_dir = r"../dataset/Published_database_FV-USM_Dec2013/Published_database_FV-USM_Dec2013/1st_session/extractedvein"

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