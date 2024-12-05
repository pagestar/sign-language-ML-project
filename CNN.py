import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Dataloader import FrameDataset
import tqdm

class FrameCNN(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super(FrameCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            nn.Conv2d(64, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.fc = torch.nn.Linear(3200, 256)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)  # (batch_size, feature_dim)
        logits = self.classifier(features)  # 分類輸出
        return features, logits  # 返回特徵和分類結果


def train_cnn(model: FrameCNN, optimizer, criterion, dataset: FrameDataset, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm.tqdm(dataset, unit="batch") as pbar:
            for idx, data in enumerate(dataset):
                imgs = dataset.get_frame_imgs(idx).cuda()  # imgs 應是 (num_frames, channels, height, width)
                _, _, label = data  # 單一標籤
                
                optimizer.zero_grad()
                
                features, logits = model(imgs)  # 獲取 CNN 特徵與分類結果
                
                # 將 label 擴展為與 logits 一致的形狀
                label_tensor = torch.tensor([label] * imgs.size(0)).cuda()
                
                loss = criterion(logits, label_tensor)  # 計算損失
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataset):.4f}")
    return model


def eval_cnn(model: FrameCNN, dataset: FrameDataset):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx)
            _, label = data

            features, logits = model(imgs)
            predicted_labels = torch.argmax(logits, dim=1)

            feats.append(features)
            labels.append(predicted_labels)
    return torch.cat(labels), torch.cat(feats)



def extract_and_combine_features(cnn_model, data_loader):
    cnn_model.eval()  # 設置為評估模式
    all_combined_features = []
    
    cnn_model.cuda()  # 模型移動到 GPU
    max_frames = 0
    
    with torch.no_grad():
        for frames, _, _ in data_loader:
            frames = frames.cuda()
            batch_size, num_frames, channels, height, width = frames.size()
            
            max_frames = max(max_frames, num_frames)  # 更新最大帧數
            
            frames_reshaped = frames.view(batch_size * num_frames, channels, height, width)
            feature_map, _ = cnn_model(frames_reshaped)
            feature_map = feature_map.view(batch_size, num_frames, -1)
            
            frames_downsampled = torch.nn.functional.adaptive_avg_pool2d(
                frames_reshaped, (32, 32))
            frames_flatten = frames_downsampled.view(batch_size, num_frames, -1)
            
            combined_input = torch.cat((frames_flatten, feature_map), dim=2)
            all_combined_features.append(combined_input)
    
    # 統一帧數
    all_combined_features = [pad_sequence(f, max_frames) for f in all_combined_features]
    
    return torch.cat(all_combined_features, dim=0)


def pad_sequence(tensor, target_len):
    current_len = tensor.size(1)  # 取得當前的時間步長
    if current_len < target_len:
        padding = torch.zeros(tensor.size(0), target_len - current_len, tensor.size(2)).cuda()
        return torch.cat([tensor, padding], dim=1)  # 在時間步維度補零
    return tensor