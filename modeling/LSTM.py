# /mnt/sdc/tnchen/matchine_learning/models/lstm_model.py

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM 多变量时间序列预测模型。

    输入:
        x: [batch_size, input_len, input_dim]
           例如 [B, 90, feature_dim]

    输出:
        y_hat: [batch_size, pred_len]
           例如短期预测 [B, 90]
           例如长期预测 [B, 365]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        pred_len: int = 90,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        # 预测头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, pred_len)
        )

    def forward(self, x):
        """
        x: [B, 90, input_dim]
        """

        # lstm_out: [B, 90, hidden_dim * num_directions]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 方案一：取最后一个时间步的输出
        # last_hidden: [B, hidden_dim * num_directions]
        last_hidden = lstm_out[:, -1, :]

        # 预测未来 pred_len 天
        y_hat = self.regressor(last_hidden)

        return y_hat


if __name__ == "__main__":
    # 简单测试
    batch_size = 16
    input_len = 90
    feature_dim = 20
    pred_len = 90

    x = torch.randn(batch_size, input_len, feature_dim)

    model = LSTMForecaster(
        input_dim=feature_dim,
        hidden_dim=128,
        num_layers=2,
        pred_len=pred_len,
        dropout=0.2
    )

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)