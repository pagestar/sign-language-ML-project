import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Dataloader import FrameDataset

class FrameClassifier(nn.Module):
    def __init__(self, num_classes, feature_dim=128):
        """
        Args:
            num_classes: 單詞類別數量（資料夾數量）。
            feature_dim: 特徵提取的維度大小。
        """
        super(FrameClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 每幀壓縮為 (feature_dim,)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)  # 將特徵映射到單詞數量的概率分佈

    def forward(self, x):
        """
        Args:
            x: 輸入影像張量，形狀為 (batch_size, channels, height, width)。
        Returns:
            features: 提取的特徵，形狀為 (batch_size, feature_dim)。
            logits: 輸出的分類概率分佈，形狀為 (batch_size, num_classes)。
        """
        features = self.features(x)  # 提取特徵
        features = features.view(features.size(0), -1)  # 展平
        logits = self.classifier(features)  # 分類層
        return features, F.softmax(logits, dim=1)  # 返回特徵矩陣和分類概率
    
def train_model(model: FrameClassifier, optimizer, criterion, dataset:FrameDataset, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx)
            _, lable = data
            optimizer.zero_grad()  # 梯度歸零
            features, logits = model(imgs)  # 同時獲取特徵與分類結果
            loss = criterion(logits, torch.tensor([lable]*len(imgs)))  # 計算分類損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新權重
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset):.4f}")

    # 保存模型
    # torch.save(model.state_dict(), "frame_classifier.pth")
    # print("模型訓練完成並已保存。")
    return model

def eval_model(model: FrameClassifier, X):
    model.eval()
    feat = []
    labels = []
    with torch.no_grad():
        for idx, data in enumerate(X):
            # imgs = dataset.get_frame_imgs(idx)
            features, logits = model(data)  # 同時獲取特徵和分類結果
            _, lable = data
            # print(logits.shape)
            predicted_labels = torch.argmax(logits, dim=1)
            # print("預測結果:", predicted_labels.numpy())
            # print("真實標籤:", lable)
            feat.append(features)
            labels.append(predicted_labels)
    return torch.tensor(np.array(labels)), torch.tensor(np.array(feat))

