import torch.nn as nn

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

'''
# 마지막 값 집어넣을때 나온 텐서 값을 피클로.
import pickle
import torch

# 주어진 텐서 생성
tensor_data = x

# Pickle 파일 이름 지정
filename = 'cnn_tensor_data.pkl'

# 텐서를 Pickle 파일로 저장
with open(filename, 'wb') as file:
    pickle.dump(tensor_data, file)
'''