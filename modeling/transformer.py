# /mnt/sdc/tnchen/matchine_learning/modeling/Transformer.py

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦位置编码。
    Transformer 本身不含时间顺序信息，因此需要加入位置编码。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # position: [max_len, 1]

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        # 如果 d_model 是奇数，需要防止维度不匹配
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len, :]

        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Transformer 多变量时间序列预测模型。

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
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        pred_len: int = 90,
        dropout: float = 0.1,
        pooling: str = "last"
    ):
        super().__init__()

        assert pooling in ["last", "mean"], "pooling 只能是 last 或 mean"

        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.pred_len = pred_len
        self.dropout = dropout
        self.pooling = pooling

        # 把原始多变量特征映射到 Transformer 的 d_model 维度
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=5000,
            dropout=dropout
        )

        # Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        # 多层 Transformer Encoder
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 输出回归头
        self.regressor = nn.Sequential(
            nn.LayerNorm(d_model),

            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(d_model // 2, pred_len)
        )

    def forward(self, x):
        """
        x: [B, 90, input_dim]
        """

        # [B, 90, input_dim] -> [B, 90, d_model]
        x = self.input_projection(x)

        # 加入位置编码
        x = self.positional_encoding(x)

        # Transformer 编码
        # enc_out: [B, 90, d_model]
        enc_out = self.encoder(x)

        # 聚合整个历史窗口的信息
        if self.pooling == "last":
            # 取最后一天的表示
            h = enc_out[:, -1, :]
        elif self.pooling == "mean":
            # 对 90 天表示做平均池化
            h = enc_out.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # 输出未来 pred_len 天
        y_hat = self.regressor(h)

        return y_hat


if __name__ == "__main__":
    batch_size = 16
    input_len = 90
    feature_dim = 20
    pred_len = 90

    x = torch.randn(batch_size, input_len, feature_dim)

    model = TransformerForecaster(
        input_dim=feature_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        pred_len=pred_len,
        dropout=0.1,
        pooling="last"
    )

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)