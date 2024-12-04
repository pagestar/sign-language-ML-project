import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import tqdm

def train_model(model, initial_lr, criterion, dataset, epochs):
    model.cuda()  # 移動模型到 GPU
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)  # 每 10 個 epoch 學習率乘以 0.9
    
    loss_record = [4.8]
    
    for epoch in range(epochs):
        total_loss = 0
        with tqdm.tqdm(enumerate(dataset), total=len(dataset)) as pbar:
            for idx, data in pbar:
                #imgs = dataset.get_frame_imgs(idx).cuda()  # 影像數據移動到 GPU
                frames, logits, label = data  # 假設 data 包含 frames 和 label
                label = label.cuda()  # 確保 label 也被移動到 GPU
                frames = frames.cuda()  # 確保 frames 也被移動到 GPU
                
                optimizer.zero_grad()  # 梯度歸零
                feature_map, logits = model(frames)  # 前向傳播，傳入 frames 和 imgs

                #print(f"Logits shape: {logits.shape}")
                #print(f"Label shape: {label.shape}")


                loss = criterion(logits, label)  # 計算損失
                loss.backward()  # 反向傳播
                optimizer.step()  # 更新權重

                total_loss += loss.item()
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        scheduler.step()  # 更新學習率
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f} ({(avg_loss - loss_record[-1]):.4f})")
        loss_record.append(avg_loss)
    
    return model, loss_record




def eval_model(model, dataset):
    model.eval()
    feat = []
    labels = []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx).cuda()  # 影像數據移動到 GPU
            features, logits = model(imgs)  # 模型前向傳播得到特徵和 logits
            
            # 確認 logits 維度是否正確
            if logits.dim() != 2:
                raise ValueError(f"Logits shape should be (batch_size, num_classes), but got {logits.shape}")
            
            # 使用 argmax 獲取每個樣本的預測類別
            predicted_labels = torch.argmax(logits, dim=1)
            
            # 如果需要保存特徵，可以取時間維度的平均作為壓縮的特徵表示
            features_mean = features.mean(dim=1)  # (batch_size, feature_dim)

            feat.append(features_mean.cpu())  # 將特徵移動到 CPU
            labels.append(predicted_labels.cpu())  # 將預測標籤移動到 CPU

    return torch.cat(labels), torch.cat(feat)



