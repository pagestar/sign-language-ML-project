import torch
from CNN import FrameCNN
from RNN import FrameRNN

class GestureModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers = 1):
        super(GestureModel, self).__init__()
        self.cnn = FrameCNN(feature_dim = input_size)
        self.rnn = FrameRNN(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, num_classes = num_classes)

    def forward(self, x):
        # print("Shape: ", x.size())  # 檢查x的形狀
        batch_size, channels, height, width = x.size()
        num_frames = 1
        x = x.view(batch_size * num_frames, channels, height, width)
        feature_map = self.cnn(x)

        feature_map = feature_map.view(batch_size, num_frames, -1)
        logits = self.rnn(feature_map)
        return feature_map,logits
    


