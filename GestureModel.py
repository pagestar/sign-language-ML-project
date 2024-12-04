import torch
from CNN import FrameCNN
from RNN import FrameRNN

class GestureModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(GestureModel, self).__init__()
        self.cnn = FrameCNN(feature_dim=input_size)  # CNN 提取特徵
        self.rnn = FrameRNN(input_size=input_size + input_size,  # 合併後的維度增加
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            num_classes=num_classes)
        self.fc = torch.nn.Linear(3200, 256)

    def forward(self, frames):
        batch_size, num_frames, channels, height, width = frames.unsqueeze(0).size()  # 原始維度
        
        # CNN 提取特徵
        frames_reshaped = frames.view(batch_size * num_frames, channels, height, width)  # (batch_size * num_frames, channels, height, width)
        feature_map = self.cnn(frames_reshaped)  # CNN 特徵輸出 (batch_size * num_frames, feature_dim)
        feature_map = feature_map.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_dim)
        
        # 合併 frames 與 feature_map
        frames_downsampled = torch.nn.functional.adaptive_avg_pool2d(frames, (32, 32))
        frames_flatten = frames_downsampled.view(batch_size, num_frames, -1)  # 將 frames 展平成 (batch_size, num_frames, channels * height * width)
        combined_input = torch.cat((frames_flatten, feature_map), dim=2)  # 在 feature 維度拼接，shape 為 (batch_size, num_frames, channels * height * width + feature_dim)

        #print(f"frames_flatten shape: {frames_flatten.shape}")  # 應該是 (batch_size, num_frames, channels * height * width)
        #print(f"feature_map shape: {feature_map.shape}")  # 應該是 (batch_size, num_frames, feature_dim)

        # 將合併後的輸入傳入 RNN
        combined_input = self.fc(combined_input)  # 全連接層輸出 (batch_size, num_frames, 256)
        logits = self.rnn(combined_input)  # RNN 輸出 logits
        #print("combined_input shape: ", combined_input.shape)
        return combined_input, logits





    


