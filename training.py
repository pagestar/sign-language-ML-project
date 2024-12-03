import torch

def train_model(model, optimizer, criterion, dataset, epochs):
    model.train()
    loss_record = []
    for epoch in range(epochs):
        total_loss = 0
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx).cuda()  # 將影像數據移動到 GPU
            _, label = data
            label = torch.tensor([label] * len(imgs)).cuda()  # 標籤移動到 GPU

            optimizer.zero_grad()  # 梯度歸零
            _, logits = model(imgs)  # CNN-RNN 前向傳播
            loss = criterion(logits, label)  # 計算損失
            loss.backward()  # 反向傳播
            optimizer.step()  # 更新權重

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataset):.4f}")
        loss_record.append(total_loss / len(dataset))
    return model, loss_record


def eval_model(model, dataset):
    model.eval()
    feat = []
    labels = []
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            imgs = dataset.get_frame_imgs(idx).cuda()  # 影像數據移動到 GPU
            features, logits = model(imgs)
            predicted_labels = torch.argmax(logits, dim=1)
            feat.append(features.cpu())  # 移回 CPU 以便後續處理
            labels.append(predicted_labels.cpu())  # 移回 CPU
    return torch.cat(labels), torch.cat(feat)


