import torch
from torch.optim import Adam
import model as m

if __name__ == "__main__":
    epoch = 500
    dropout = 0.3
    size = 12

    # Change the model type for experiment
    model_type = 'LSTM'

    if model_type == 'LSTM':
        model = m.LSTM(size=size, )
    elif model_type == 'GRU':
        model = m.GRU()
    elif model_type == 'Attention':
        model = m.LSTM()

    # Optimizer
    optimizer = Adam()
