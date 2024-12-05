import torch
from CNN import FrameCNN
from RNN import FrameRNN

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






    


