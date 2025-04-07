import torch
from torch.nn import LSTM, BatchNorm1d, Identity, Linear, Module


class HurricaneRNN(Module):
    def __init__(self, name, series_length, dropout=0.0):
        super().__init__()
        self.name = name
        self.dropout = dropout
        self.hidden_size = 256
        self.bn = BatchNorm1d(series_length)
        self.lstm1 = LSTM(
            input_size=2,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.dense = Linear(
            in_features=self.hidden_size * series_length + 2 * series_length,
            out_features=256,
        )
        self.out = Linear(in_features=256, out_features=2)
        self.resblock = Identity()

    def forward(self, x):
        x1 = self.resblock(x)
        x, (h1, c1) = self.lstm1(x)
        x = torch.cat((x, x1), dim=2)
        x = self.dense(x.view(len(x), -1))
        x = self.out(x)
        return x
