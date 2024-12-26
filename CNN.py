import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Dataloader import FrameDataset
import tqdm
import random

'''
Update: adding more layers to the CNN architecture
You can modify it.
'''
class FrameCNN(nn.Module):
    def __init__(self, feature_dim=128, num_classes=10):
        super(FrameCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),  # 引入 Layer Normalization
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=5, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()
    loss_lst = []
    test_acc = []
    min_loss, patience_count = 10, 0
    # random choose 15 videos from 39 videosto test
    test_lst = random.sample(range(39), 15)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        with tqdm.tqdm(dataset, unit="batch") as pbar:
            for idx, data in enumerate(dataset):
                imgs = dataset.get_frame_imgs(idx).cuda()  # imgs 是 (num_frames, channels, height, width)
                _, _, label = data  # 单一标签
                
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():  # 开启混合精度
                    features, logits = model(imgs)  # 获取 CNN 特征与分类结果
                    # 将 label 扩展为与 logits 一致的形状
                    label_tensor = torch.tensor([label] * imgs.size(0), dtype=torch.long).cuda()
                    loss = criterion(logits, label_tensor)  # 计算损失
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        avg_loss = total_loss / len(dataset)
        scheduler.step(avg_loss)

        if avg_loss < min_loss:
            min_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count == 20:
                print("Early stopping...")
                break

        # 每个epoch后评估一次
        test_acc.append(eval_cnn(model, dataset, test_lst))

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Testing ACC: {test_acc[-1]:.4f}, Patience: {patience_count}/20")
        loss_lst.append(avg_loss)

    return model, loss_lst, test_acc

def eval_cnn(model: FrameCNN, dataset: FrameDataset, test_lst):
    test_image, _, label = dataset[0]  # frame, logits, label
    test_image = test_image[40].unsqueeze(0)  
    test_image = test_image.cuda()
    label = label.cuda()

    correct = 0

    
    for idx in test_lst:  
        test_image, _, label = dataset[idx]
        test_image = test_image[10].unsqueeze(0).cuda()  
        
        with torch.no_grad():
            _, logits = model(test_image)
            predicted_label = torch.argmax(logits, dim=1).item()

        if predicted_label == label.item():
            correct += 1

    accuracy = correct / 15
    return accuracy



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

    correct = 0

    for idx in range(30):  
        test_image, _, label = dataset[idx]
        test_image = test_image[10].unsqueeze(0).cuda()  
        
        with torch.no_grad():
            _, logits = model(test_image)
            predicted_label = torch.argmax(logits, dim=1).item()
            
        print(f"Predicted: {predicted_label}, Actual: {label.item()}")

        if predicted_label == label.item():
            correct += 1

    print(f"Accuracy: {(correct / 30):.4f}")
    
'''
Combnine the feature map and the frame features to get the final input for the LSTM model.
Note that the output combines all the frames and maps in a single word (video).

Suggestions:
    1. There might be somethign to fix.
    2. We pad the sequences with zeros to make them all the same length (of total frames). Consider implemtenting a mask for RNN.
    3. Do we need to use the original frames?
'''
def extract_logits(cnn_model, data_loader):
    cnn_model.eval()  
    all_logits = []
    
    cnn_model.cuda() 

    max_frames = 0
    
    with torch.no_grad():
        for frames, _, _ in data_loader:
            frames = frames.cuda()
            batch_size, num_frames, channels, height, width = frames.size()
            max_frames = max(max_frames, num_frames)
            
            frames_reshaped = frames.view(batch_size * num_frames, channels, height, width)
            
            # 获取 CNN 的特征映射（logits）
            feature_map, _ = cnn_model(frames_reshaped)
            
            # Reshape feature_map 为 (batch_size, num_frames, -1)
            feature_map = feature_map.view(batch_size, num_frames, -1)
            
            all_logits.append(feature_map)
    
    # 合并所有帧的特征到一个大的 tensor 中
    all_logits = [pad_sequence(logits, max_frames) for logits in all_logits]
    
    return torch.cat(all_logits, dim=0)



def pad_sequence(tensor, target_len):
    current_len = tensor.size(1)  # 取得當前的時間步長
    if current_len < target_len:
        padding = torch.zeros(tensor.size(0), target_len - current_len, tensor.size(2)).cuda()
        return torch.cat([tensor, padding], dim=1)  # 在時間步維度補零
    return tensor
