import torch
from RNN import FrameRNN
from torch.optim.lr_scheduler import StepLR
import tqdm

class GestureModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(GestureModel, self).__init__()
        self.rnn = FrameRNN(input_size=input_size,  # CNN 提取的特徵和降維後 frames 合併後的維度
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            num_classes=num_classes)

    def forward(self, combined_features):
        """
        combined_features: (batch_size, num_frames, combined_feature_dim)
        """
        logits = self.rnn(combined_features)  # 傳入 RNN 的輸入是 CNN 和降維 frames 的拼接結果
        return logits

'''
Train the model. The learning rate is decayed by a factor of 0.9 every 30 epochs.
Note that the dataloader contains "all_combined_features"
'''
def train_model(model, criterion, optimizer, dataset, epochs):
    model.cuda()  # 将模型移到 GPU
    
    model.train()  # 确保模型在训练模式下
    scheduler = StepLR(optimizer, step_size=30, gamma=0.9)  # 每30个epoch降低学习率
    
    for epoch in range(epochs):
        total_loss = 0
        with tqdm.tqdm(enumerate(dataset), total=len(dataset)) as pbar:
            for idx, data in pbar:
                frames, label = data
                label = label.cuda()  
                frames = frames.cuda()  
                
                optimizer.zero_grad()  
                logits = model(frames)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                
                # 更新 tqdm 描述
                pbar.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        scheduler.step()  # 在每个 epoch 结束时更新学习率
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return model


'''
Evaluate the model.
'''
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




    


