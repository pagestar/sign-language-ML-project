import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from Dataloader import FrameDataset, VideoDataAugmentation
from GestureModel import GestureModel
from torch.utils.data import DataLoader, TensorDataset
from training import eval_model, train_model
from torch.nn.utils.rnn import pad_sequence
from CNN import FrameCNN, train_cnn, eval_cnn, extract_and_combine_features
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
    num_layers = 10
    learning_rate = 0.0001
    epochs = 1000

    transform = VideoDataAugmentation()

    # 2. 數據集與數據加載
    dataset = FrameDataset(root_dir=root_dir, transform=transform)  # 使用自定義的FrameDataset類加載數據
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)  # 定義數據載入器

    cnn = FrameCNN(feature_dim=128).cuda()  # 定義 CNN，輸出特徵維度 128
    rnn = GestureModel(input_size=128 + 3 * 32 * 32,  # 合併 feature_map 和 frames_flatten 的維度
                    hidden_size=hidden_dim, num_classes=num_classes, num_layers=num_layers).cuda()

    # 定義優化器和損失函數
    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # 1. 訓練 CNN
    print("開始訓練 CNN...")
    #train_cnn(cnn, cnn_optimizer, criterion, dataset, num_epochs=100)
    #eval_cnn(cnn, dataset)
    #torch.save(cnn.state_dict(), "trained_cnn.pth")  # 保存訓練好的 CNN

    # 2. 提取 CNN 特徵並合併降維 frames
    cnn.load_state_dict(torch.load("trained_cnn.pth"))  # 加載訓練好的 CNN
    combined_features = extract_and_combine_features(cnn, dataloader)

    # 3. 准備 RNN 訓練
    labels = torch.tensor(dataset.labels, dtype=torch.long)  # 確保 labels 也與訓練數據匹配
    rnn_dataset = TensorDataset(combined_features, labels)
    rnn_loader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)

    # 4. 訓練 RNN
    print("開始訓練 RNN...")
    train_model(rnn, criterion, rnn_optimizer, rnn_loader, epochs)

    # 5. 評估模型
    print("開始評估模型...")
    labels = eval_model(rnn, rnn_loader)
    print("評估結果：")
    #print("特徵向量維度：", features.shape)
    #print("模型準確率：", (labels == labels.max(1)[1]).sum().item() / labels.shape[0])
    print("Augmentation count:", transform.augmentation_count)
    print_result(labels, dataset)

    '''
    # 6. 繪製訓練過程圖
    plt.plot(loss_list[1:])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Hidden dim: {hidden_dim}, Learning rate: {learning_rate}, layers: {num_layers}")
    plt.show()
    '''


if __name__ == "__main__":
    main()
