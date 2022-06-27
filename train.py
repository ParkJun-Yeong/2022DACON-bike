import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# import torchsummary
from torchinfo import summary
import torch.nn.init as weight_init
from ray import ray

from datetime import datetime
from tqdm import tqdm
import os

import model as model_list
from dataset import BikeDataset, collate_fn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def nmse(true, pred):
    score = torch.mean(torch.abs(true-pred) / true)
    # score = np.mean(np.abs(true-pred) / true)
    return score

def train_loop(dataloader, model, optimizer, epochs):
    print("device: ", device)
    writer = SummaryWriter()
    model.train()

    print("========== Train..... ==========")
    iter = 0
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        loss_history = []
        for i, (X, tgt) in tqdm(enumerate(dataloader), desc="Train....."):
            tgt = torch.tensor(tgt).to(device, dtype=torch.float32)
            pred = model(X.to(device)).to(device, dtype=torch.float32)
            loss = loss_fn(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())
            # print("====Epoch [", epoch, "][", i, "] Loss: ", loss)
            writer.add_scalar("Loss per iter/train", loss, iter)
            iter += 1

        loss_history = torch.tensor(loss_history)
        mean_loss = torch.mean(loss_history)
        print("====Epoch [", epoch, "] Mean Loss: ", mean_loss, "=====")
        writer.add_scalar("Mean Loss per epoch/train", mean_loss, epoch)

    writer.flush()
    writer.close()

    saved_model_dir = r'./saved_model'
    now = datetime.now()
    torch.save(model, os.path.join(saved_model_dir, "saved_model" + now.strftime("%Y-%m-%d-%H-%M") + ".pt"))


# 10-cross validation


if __name__ == "__main__":
    epochs = 3000
    size = 15
    hidden_size = size
    # num_layers = [4, 12, 24]
    num_layers = [1]
    # stack = 3
    stack = 1
    dropout = 0.1
    # batch_size = 384
    batch_size = 32
    learning_rate = 10
    # weight_decay = 1e-5
    # weight_decay = 2e-5

    # Change the model type for experiment
    model_type = 'GRU'

    if model_type == 'LSTM':
        model = model_list.LSTM(size, hidden_size, num_layers, stack, dropout)
    elif model_type == 'GRU':
        model = model_list.GRU(size, hidden_size, num_layers, stack, dropout)
    elif model_type == 'Attention':
        model = model_list.LSTM()

    model.to(device)

    dict = {}
    for name, param in model.named_parameters():
        weight_init.normal(param)
        dict[name] = param

    # torchsummary로 RNN 계열 모델을 요약하려고 하면 입력값사이즈 인자를 튜플로 인식해서 에러 발생
    # print(torchsummary.summary(model, (1, 16)))

    # 이 문제를 해결한 from torchinfo import summary 사용
    print(summary(model, input_size=(batch_size, size)))


    # setting
    train_dataset = BikeDataset(train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loop(dataloader=train_dataloader, model=model, optimizer=optimizer, epochs=epochs)
