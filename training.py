import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
import tqdm




def train_rnn(rnn_model, features, labels, criterion, optimizer, num_epochs):
    rnn_model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = rnn_model(features)  # 用 CNN 提取並拼接的特徵
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

def train_model(model, criterion, optimizer, dataset, epochs):
    model.cuda()  # 移動模型到 GPU
    model.train()
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.9)  # 每 10 個 epoch 學習率乘以 0.9
    scaler = GradScaler()
    loss_record = [4]
    
    for epoch in range(epochs):
        total_loss = 0
        with tqdm.tqdm(enumerate(dataset), total=len(dataset)) as pbar:
            for idx, data in pbar:
                #imgs = dataset.get_frame_imgs(idx).cuda()  # 影像數據移動到 GPU
                frames, label = data  # 假設 data 包含 frames 和 label
                label = label.cuda()  # 確保 label 也被移動到 GPU
                frames = frames.cuda()  # 確保 frames 也被移動到 GPU
                
                optimizer.zero_grad()  # 梯度歸零
                logits = model(frames)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        scheduler.step()  # 更新學習率
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f} ({(avg_loss - loss_record[-1]):.4f})")
        loss_record.append(avg_loss)
    
    return model, loss_record


def eval_model(model, dataset):
    model.eval()
    labels = []
    
    model.cuda()  # 確保模型在 GPU 上
    
    with torch.no_grad():
        for frame, label in dataset:

            frame = frame.cuda()  
            label = label.cuda()  

            # 前向传播通过 CNN 提取特征和 RNN 进行分类
            logits = model(frame)  # 假设模型能够处理该输入并返回 logits
            
            # 确保 logits 的形状是 (batch_size, num_classes)
            if logits.dim() != 2:
                raise ValueError(f"Logits shape should be (batch_size, num_classes), but got {logits.shape}")
            
            # 使用 argmax 获取每个样本的预测类别
            predicted_labels = torch.argmax(logits, dim=1)  # (batch_size,)
        
            labels.append(predicted_labels.cpu())  # 将预测标签移动到 CPU 并存入列表

    return torch.cat(labels)  # 返回所有预测标签





