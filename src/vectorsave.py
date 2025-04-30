import pickle
import torch
from torch.utils.data import Dataset

# pickle data load
with open('./data/train_data.pkl', 'rb') as f:
    data = pickle.load(f)
   # print(data)

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # 예시: 이미지와 레이블이 tuple 형태로 저장되어 있다고 가정
        image, label = sample[0], sample[1]

        if self.transform:
            image = self.transform(image)

        return image, label