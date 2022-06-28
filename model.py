import torch
import torch.nn as nn

"""
Argument
- size: input size(dimension) = # of features
- stack: # of lstm stacks
- 
"""
class LSTM(nn.Module):
    def __init__(self, size, hidden_size, num_layers, stack, dropout):
        super(LSTM, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack = stack
        self.dropout = dropout
        self.module = nn.ModuleList([nn.LSTM(input_size=self.size, hidden_size=self.hidden_size,
                                             num_layers=self.num_layers[i], batch_first=True, dropout=self.dropout) for i in range(self.stack)])
        self.fc = nn.Linear(in_features=self.size, out_features=1)

    def forward(self, x):
        # x (batch, 1, size) = (batch, 1, 16)
        x = x.unsqueeze(1)

        # Initial hn and cn are zero
        # hn, cn (num_layers, batch, hidden_size)
        x , (hn, cn) = self.module[0](x)

        # # (4, batch, 16) -> (12, batch, 16)
        # hn = torch.concat([hn, hn, hn], dim=0)
        # cn = torch.concat([cn, cn, cn], dim=0)
        # x, (hn, cn) = self.module[1](x, (hn, cn))

        # # (12, batch, 16) -> (24, batch, 16)
        # hn = torch.concat([hn, hn], dim=0)
        # cn = torch.concat([cn, cn], dim=0)
        # x, (hn, cn) = self.module[2](x, (hn, cn))

        output = x

        y = self.fc(output)

        # to scalar
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        return y

class GRU(nn.Module):
    def __init__(self, size, hidden_size, num_layers, stack, dropout):
        super(GRU, self).__init__()
        self.size = size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.stack = stack
        self.dropout = dropout
        self.module = nn.ModuleList([nn.GRU(input_size=self.size, hidden_size=self.hidden_size,
                                             num_layers=self.num_layers[i], batch_first=True, dropout=self.dropout) for i in range(self.stack)])
        self.fc = nn.Linear(in_features=self.size, out_features=1)

    def forward(self, x):
        # x (batch, 1, size) = (batch, 1, 16)
        x = x.unsqueeze(1)

        # Initial hn and cn are zero
        # hn, cn (num_layers, batch, hidden_size)
        x, hn = self.module[0](x)

        # # (4, batch, 16) -> (12, batch, 16)
        # hn = torch.concat([hn, hn, hn], dim=0)
        # cn = torch.concat([cn, cn, cn], dim=0)
        # x, (hn, cn) = self.module[1](x, (hn, cn))

        # # (12, batch, 16) -> (24, batch, 16)
        # hn = torch.concat([hn, hn], dim=0)
        # cn = torch.concat([cn, cn], dim=0)
        # x, (hn, cn) = self.module[2](x, (hn, cn))

        output = x

        y = self.fc(output)

        # to scalar
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        return y


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # self.fc1 = nn.Linear(14, 10)
        # self.fc2 = nn.Linear(10, 5)
        # self.fc3 = nn.Linear(5, 1)
        self.act_fuc = nn.Sigmoid()
        self.fc1 = nn.Linear(13, 5)
        self.fc2 = nn.Linear(5, 1)



    def forward(self, x):
        y = self.fc1(x)
        y = self.act_fuc(y)
        y = self.fc2(y)
        # y = self.act_fuc(y)
        # y = self.fc3(y)
        return y