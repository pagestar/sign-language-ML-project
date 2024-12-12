import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Dataloader import FrameDataset
import tqdm

'''
Update: adding more layers to the CNN architecture
You can modify it.
'''
class FrameCNN(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super(FrameCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)  # (batch_size, feature_dim)
        logits = self.classifier(features)  # 分類輸出
        return features, logits  # 返回特徵和分類結果

'''
Update: modify the scheduler
The learning rate will be reduced by the factor of 0.5 if the loss does not improve for 5 epochs.
You can modify it.
'''
def train_cnn(model: FrameCNN, optimizer, criterion, dataset: FrameDataset, num_epochs):
    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    loss_lst = []
    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm.tqdm(dataset, unit="batch") as pbar:
            for idx, data in enumerate(dataset):
                imgs = dataset.get_frame_imgs(idx).cuda()  # imgs 應是 (num_frames, channels, height, width)
                _, _, label = data  # 單一標籤
                
                optimizer.zero_grad()
                
                features, logits = model(imgs)  # 獲取 CNN 特徵與分類結果
                
                # 將 label 擴展為與 logits 一致的形狀
                label_tensor = torch.tensor([label] * imgs.size(0), dtype=torch.long).cuda()

                
                loss = criterion(logits, label_tensor)  # 計算損失
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = total_loss / len(dataset)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        loss_lst.append(avg_loss)
    return model, loss_lst

'''
Not used for now.
'''
def eval_cnn(model: FrameCNN, dataset: FrameDataset):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx)
            _, label, _ = data

            features, logits = model(imgs)
            predicted_labels = torch.argmax(logits, dim=1)

            feats.append(features)
            labels.append(predicted_labels)
    return torch.cat(labels), torch.cat(feats)

'''
Test the CNN performance on a few test samples.
Here we use the 11th frame of first 10 videos as test samples.
You can modify it.
'''
def test_CNN(model: FrameCNN, dataset: FrameDataset):
    test_image, _, label = dataset[0]  # frame, logits, label
    test_image = test_image[40].unsqueeze(0)  
    test_image = test_image.cuda()
    label = label.cuda()

    for idx in range(10):  
        test_image, _, label = dataset[idx]
        test_image = test_image[10].unsqueeze(0).cuda()  
        
        with torch.no_grad():
            _, logits = model(test_image)
            predicted_label = torch.argmax(logits, dim=1).item()
            
        print(f"Predicted: {predicted_label}, Actual: {label.item()}")
        

'''
Combnine the feature map and the frame features to get the final input for the LSTM model.
Note that the output combines all the frames and maps in a single word (video).

Suggestions:
    1. There might be somethign to fix.
    2. We pad the sequences with zeros to make them all the same length (of total frames). Consider implemtenting a mask for RNN.
    3. Do we need to use the original frames?
'''
def extract_and_combine_features(cnn_model, data_loader):
    cnn_model.eval()  
    all_combined_features = []
    
    cnn_model.cuda() 
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
    
    
    all_combined_features = [pad_sequence(f, max_frames) for f in all_combined_features]
    
    return torch.cat(all_combined_features, dim=0)


def pad_sequence(tensor, target_len):
    current_len = tensor.size(1)  # 取得當前的時間步長
    if current_len < target_len:
        padding = torch.zeros(tensor.size(0), target_len - current_len, tensor.size(2)).cuda()
        return torch.cat([tensor, padding], dim=1)  # 在時間步維度補零
    return tensor
