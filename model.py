import torch
import torch.nn as nn

"""
Argument
- size: input size(dimension) = # of features
- stack: # of lstm stacks
- 
"""
class LSTM(nn.Module):
    def __init__(self, size, stack, dropout, ):
        super(LSTM, self).__init__()
        self.size = size
        self.hidden_size = size
        self.stack = stack
        self.module = nn.ModuleList([nn.LSTM(input_size=size, hidden_size=self.hidden_size,
                                             batch_first=True, dropout=dropout) for _ in range(stack)])
        self.fc = nn.Linear(in_features=size, out_features=1)

    def forward(self, x):
        hn = 0
        cn = 0
        for i in range(self.stack):
            output, (hn, cn) = self.module[i](x, (hn, cn))

        y = self.fc(output)

        return y





class GRU(nn.Module):
    def __init__(self):
        super(GRU, self)