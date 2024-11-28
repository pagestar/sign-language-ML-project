import torch
import torch.optim as optim
import torch.nn as nn


 # 定義優化器

def train(model, train_loader, num_epochs=10):
    """
    訓練 GestureRNN 模型

    Args:
        model (nn.Module): 訓練的模型。
        train_loader (DataLoader): 訓練數據的 DataLoader。
        criterion (nn.Module): 損失函數。
        optimizer (optim.Optimizer): 優化器。
        device (torch.device): 運行的設備（CPU 或 GPU）。
        num_epochs (int): 訓練的迭代次數（epoch）。
    """
    criterion = nn.CrossEntropyLoss()  # 定義損失函數
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    model.train()  # 設置模型為訓練模式
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()  # 移動數據到 GPU

            # 清零梯度
            optimizer.zero_grad()

            # 預測
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)

            # 反向傳播
            loss.backward()

            # 更新權重
            optimizer.step()

            running_loss += loss.item()

        # 每個 epoch 結束時打印損失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")


def evaluate(model, test_loader):
    """
    評估模型的表現

    Args:
        model (nn.Module): 訓練好的模型。
        test_loader (DataLoader): 測試數據的 DataLoader。
        criterion (nn.Module): 損失函數。
        device (torch.device): 運行的設備（CPU 或 GPU）。
    """
    model.eval()  # 設置模型為評估模式
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()  # 定義損失函數
    with torch.no_grad():  # 評估階段不需要計算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            # 預測
            outputs = model(inputs)

            # 計算損失
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 計算準確度
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Loss: {running_loss / len(test_loader)}")
    print(f"Accuracy: {100 * correct / total}%")

