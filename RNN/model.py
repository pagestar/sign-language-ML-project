import torch
import torch.nn as nn

class GestureRNN(nn.Module):
    """
    GestureRNN 模型的框架。這個模型包含一個 RNN 層，然後是用於分類的全連接層。
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化 GestureRNN 模型的架構。

        Args:
            input_size (int): 每幀的特徵數（例如，關鍵點的 x, y, z 座標數量）。
            hidden_size (int): RNN 隱藏層的大小。
            output_size (int): 類別數量（手勢的種類數量）。
        """
        super(GestureRNN, self).__init__()

        # 初始化 RNN 層
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)

        # 初始化全連接層（FC 層）進行分類
        self.fc = nn.Linear(hidden_size, output_size)

        # 儲存 hidden_size 和 output_size 以便進行 assert 檢查
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        """
        前向傳遞函數，將數據輸入 RNN 並返回分類結果。

        Args:
            x (Tensor): 輸入數據，形狀應該是 (batch_size, sequence_length, input_size)。

        Returns:
            Tensor: RNN 層的輸出，形狀是 (batch_size, output_size)，即每個樣本的分類結果。
        """
        # 驗證輸入形狀
        # print(f"Input shape: {x.shape}")
        batch_size, sequence_length, _ = x.shape

        # 向前傳遞數據，通過 RNN 層
        out, _ = self.rnn(x)
        assert out.shape == (batch_size, sequence_length, self.hidden_size), \
            f"RNN output shape should be {(batch_size, sequence_length, self.hidden_size)}, but got {out.shape}"
        
        # 從 RNN 輸出中取最後一幀的隱藏狀態
        last_hidden_state = out[:, -1, :]
        assert last_hidden_state.shape == (batch_size, self.hidden_size), \
            f"Last hidden state shape should be {(batch_size, self.hidden_size)}, but got {last_hidden_state.shape}"    

        # 使用全連接層進行分類
        out = self.fc(last_hidden_state)
        assert out.shape == (batch_size, self.output_size), \
            f"Output shape should be {(batch_size, self.output_size)}, but got {out.shape}"

        return out
