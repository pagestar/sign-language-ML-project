import torch
from torch.utils.data import DataLoader, TensorDataset

from convert_data import process_videos
from model import GestureRNN
from train_model import train

def pad_features(input_tensors, target_size):
    padded_tensors = []
    for tensor in input_tensors:
        padding_size = target_size - tensor.shape[2]
        if padding_size > 0:
            padding = torch.zeros((tensor.shape[0], tensor.shape[1], padding_size))
            padded_tensor = torch.cat((tensor, padding), dim=2)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)
    return padded_tensors

def main():
    video_paths = [
        "data\晚餐_1.mp4", "data\晚餐_2.MP4", "data\校長_1.mp4", "data\校長_2.MP4",
        "data\桌子_1.mp4", "data\桌子_2.mp4", "data\風_1.mp4", "data\風_2.MP4",
        "data\素食_A_1.mp4", "data\素食_A_2.MP4", "data\素食_B_1.mp4", "data\素食_B_2.MP4",
    ]

    rnn_inputs = process_videos([path.strip() for path in video_paths])

    max_feature_size = max(tensor.shape[2] for tensor in rnn_inputs)
    padded_inputs = pad_features(rnn_inputs, max_feature_size)

    inputs_list = []
    labels_list = []

    id = 0

    for i, rnn_input in enumerate(padded_inputs):
        inputs_list.append(rnn_input)
        labels_list.extend([id] * rnn_input.shape[0])  # 擴展標籤使其符合幀數
        if i % 2 == 1:
            id += 1

    # 將所有影片數據和標籤整合
    inputs = torch.cat(inputs_list, dim=0).float()
    labels = torch.tensor(labels_list, dtype=torch.long)

    print("Total input shape:", inputs.shape)
    print("Total label shape:", labels.shape)

    # 建立 DataLoader
    dataset = TensorDataset(inputs, labels)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 定義模型超參數
    input_size = max_feature_size
    hidden_size = 128
    output_size = len(video_paths)

    # 初始化模型
    model = GestureRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model = model.cuda()

    # 訓練模型
    num_epochs = 50
    train(model, train_loader, num_epochs)

if __name__ == "__main__":
    main()
