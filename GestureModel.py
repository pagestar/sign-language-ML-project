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
        batch_size, num_frames, channels, height, width = frames.size()  # 原始維度
        
        # CNN 提取特徵
        frames_reshaped = frames.view(batch_size * num_frames, channels, height, width)  # (batch_size * num_frames, channels, height, width)
        feature_map = self.cnn(frames_reshaped)  # CNN 特徵輸出 (batch_size * num_frames, feature_dim)
        feature_map = feature_map.view(batch_size, num_frames, -1)  # (batch_size, num_frames, feature_dim)
        
        # 對每個時間步進行自適應池化
        frames_downsampled = torch.nn.functional.adaptive_avg_pool2d(frames.view(batch_size * num_frames, channels, height, width), (32, 32))  # (batch_size * num_frames, channels, 32, 32)
        
        # 將 frames_downsampled 還原回原始形狀 (batch_size, num_frames, channels * 32 * 32)
        frames_flatten = frames_downsampled.view(batch_size, num_frames, -1)  # (batch_size, num_frames, channels * 32 * 32)
        
        # 合併 frames_flatten 和 feature_map
        combined_input = torch.cat((frames_flatten, feature_map), dim=2)  # (batch_size, num_frames, channels * 32 * 32 + feature_dim)
        
        # 將合併後的輸入傳入全連接層
        combined_input = self.fc(combined_input)  # 全連接層輸出 (batch_size, num_frames, 256)
        
        # 傳入 RNN 進行處理
        logits = self.rnn(combined_input)  # RNN 輸出 logits
        
        return combined_input, logits






    


