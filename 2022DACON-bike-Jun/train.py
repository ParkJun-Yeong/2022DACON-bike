import torch
from torch.optim import Adam
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
# import torchsummary
# import torch.nn.init as weight_init
# from ray import ray
# from dataset import BikeDataset, collate_fn

from datetime import datetime
from tqdm import tqdm
import os
import warnings

import model as model_list
from preprocess import get_train_test

warnings.simplefilter(action='ignore', category=[DeprecationWarning]) # DeprecationWarning 제거
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def NMAE(true, pred):
    true = true.detach().numpy()
    score = np.mean(np.abs(true-pred) / true)
    return score


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
    batch_size = 1
    learning_rate = 1
    # weight_decay = 1e-5
    # weight_decay = 2e-5

    train = True

    # Change the model type for experiment
    model_type = 'LSTM'

    if model_type == 'LSTM':
        model = model_list.LSTM(size, hidden_size, num_layers, stack, dropout)
    elif model_type == 'GRU':
        model = model_list.GRU(size, hidden_size, num_layers, stack, dropout)
    elif model_type == 'Attention':
        model = model_list.LSTM()

    model.to(device)

    # dict = {}
    # for name, param in model.named_parameters():
    #     weight_init.normal(param)
    #     dict[name] = param

    X, X_year, y, X_test, X_test_year, y_test = get_train_test()
    X = torch.tensor(X, dtype=torch.float32)
    X_year = torch.tensor(X_year, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test_year = torch.tensor(X_test_year, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # torchsummary로 RNN 계열 모델을 요약하려고 하면 입력값사이즈 인자를 튜플로 인식해서 에러 발생
    # print(torchsummary.summary(model, (1, 16)))

    # 이 문제를 해결한 from torchinfo import summary 사용
    print(summary(model, input_size=(batch_size, size)))


    # setting
    # train_dataset = BikeDataset(train=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if train:
        print("device: ", device)
        writer = SummaryWriter()
        model.train()

        print("========== Train..... ==========")
        iter = 0
        loss_fn = torch.nn.MSELoss()

        for epoch in range(epochs):
            loss_history = []
            for i, (x, tgt) in tqdm(enumerate(zip(X_year, y))):
                tgt = tgt.to(device)
                pred = model(x.to(device)).to(device)
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
        torch.save(model.state_dict(),
                   os.path.join("./saved_model/saved_model_lstm_220629.pt"))

    if not train:
        model.load_state_dict(torch.load("./saved_model/saved_model_lstm_220629.pt"))

    model.eval()

    pred = model(X_year)
    pred = pred.detach().numpy()
    print("train nmae: ", NMAE(y, pred))

    plt.plot(range(len(X_year)), y)
    plt.plot(range(len(X_year)), pred)
    plt.show()

    pred = model(X_test)
    pred = pred.detach().numpy()
    print("test nmae: ", NMAE(y_test, pred))

    plt.plot(range(len(X_test)), y_test)
    plt.plot(range(len(X_test)), pred)
    plt.show()

    # train_loop(dataloader=train_dataloader, model=model, optimizer=optimizer, epochs=epochs)
