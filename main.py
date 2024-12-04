import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataloader import FrameDataset, VideoDataAugmentation
from GestureModel import GestureModel
from training import train_model, eval_model

def print_result(labels, dataset):
    labels = labels.cpu().numpy()
    for i in range(len(labels)):
        print(f"Video {i+1}: Predicted = {dataset.label2word[labels[i]]}, Actual = {dataset.label2word[dataset.labels[i]]}")


def main():
    # 1. 參數設定
    root_dir = "data"  
    num_classes = 113
    feature_dim = 128
    hidden_dim = 128  
    num_layers = 2  
    learning_rate = 0.0002
    epochs = 1

    transform = VideoDataAugmentation()

    # 2. 數據集與數據加載
    dataset = FrameDataset(root_dir=root_dir, transform=transform)  # 使用自定義的FrameDataset類加載數據

    # 3. 模型、優化器、損失函數初始化
    model = GestureModel(input_size=feature_dim, hidden_size=hidden_dim, num_classes=num_classes, num_layers=num_layers)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 適用於多類別分類

    # 4. 訓練模型
    print("開始訓練模型...")
    model, loss_list = train_model(model, learning_rate, criterion, dataset, epochs)

    # 5. 評估模型
    print("開始評估模型...")
    labels, features = eval_model(model, dataset)
    print("評估結果：")
    print("特徵向量維度：", features.shape)
    #print("模型準確率：", (labels == labels.max(1)[1]).sum().item() / labels.shape[0])
    print("Augmentation count:", transform.augmentation_count)
    print_result(labels, dataset)

    # 6. 繪製訓練過程圖
    plt.plot(loss_list[1:])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Hidden dim: {hidden_dim}, Learning rate: {learning_rate}, layers: {num_layers}")
    plt.show()

    torch.save(model, 'model.pth')


if __name__ == "__main__":
    main()
