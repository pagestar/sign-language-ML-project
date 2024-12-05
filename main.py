import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataloader import FrameDataset, VideoDataAugmentation
from GestureModel import GestureModel
from torch.utils.data import DataLoader
from training import train_model, eval_model
from torch.nn.utils.rnn import pad_sequence
import warnings
import os

def print_result(labels, dataset):
    labels = labels.cpu().numpy()
    for i in range(len(labels)):
        print(f"Video {i+1}: Predicted = {dataset.label2word[labels[i]]}, Actual = {dataset.label2word[dataset.labels[i]]}")

def collate_fn(batch):
    frames = [item[0] for item in batch]
    logits = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    # 填充 frames，使每個樣本的長度一致
    frames = pad_sequence(frames, batch_first=True, padding_value=0)

    # 如果 logits 的維度不一致，您也可以考慮對 logits 進行填充或處理
    logits = pad_sequence(logits, batch_first=True, padding_value=0)  # 如果 logits 是序列數據，可以用這行
    # 如果 logits 只是單一的標籤，則可以直接堆疊
    # logits = torch.stack(logits)

    labels = torch.tensor(labels, dtype=torch.long)
    
    return frames, logits, labels


def main():

    warnings.filterwarnings("ignore", category=UserWarning, message="Feedback manager requires a model with a single signature inference")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 1. 參數設定
    root_dir = "data"  
    num_classes = 10
    feature_dim = 128
    hidden_dim = 256
    num_layers = 4
    learning_rate = 0.0005
    epochs = 1500

    transform = VideoDataAugmentation()

    # 2. 數據集與數據加載
    dataset = FrameDataset(root_dir=root_dir, transform=transform)  # 使用自定義的FrameDataset類加載數據
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)  # 定義數據載入器

    # 3. 模型、優化器、損失函數初始化
    model = GestureModel(input_size=feature_dim, hidden_size=hidden_dim, num_classes=num_classes, num_layers=num_layers)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # 適用於多類別分類

    # 4. 訓練模型
    print("開始訓練模型...")
    model, loss_list = train_model(model, learning_rate, criterion, dataloader, epochs)

    torch.save(model, 'model.pth')

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

    


if __name__ == "__main__":
    main()
