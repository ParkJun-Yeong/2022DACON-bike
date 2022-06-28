import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.optim import Adam
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from model import LinearModel
from preprocess import get_train_test

def NMAE(true, pred):
    true = true.detach().numpy()
    score = np.mean(np.abs(true-pred) / true)
    return score


if __name__ == "__main__":
    loss_fuc = nn.MSELoss()
    epochs = 3000
    lr = 0.1

    train = True

    model = LinearModel()

    X, X_year, y, X_test, X_test_year, y_test = get_train_test()
    X = torch.tensor(X, dtype=torch.float32)
    X_year = torch.tensor(X_year, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test_year = torch.tensor(X_test_year, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 모델 summary
    print(summary(model, input_size=(1, 13)))

    # nn.init.xavier_normal_(model.weights, gain=nn.init.calculate_gain('relu'))

    # # 정규분포로 파라미터 초기화
    # dict = {}
    # for name, param in model.named_parameters():
    #     weight_init.normal(param)
    #     dict[name] = param

    # model.apply(init_weights)

    optimizer = Adam(model.parameters(), lr=lr)
    mean_loss = []

    if train:
        print("=====train=====")
        for epoch in trange(epochs):
            loss_history = []

            for x, tgt in zip(X, y):
                model.train()

                pred = model(x)

                #         print(pred.shape)
                #         print(tgt.shape)
                loss = loss_fuc(pred, tgt.unsqueeze(0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())
            mean_loss.append(np.mean(loss_history))

        torch.save(model.state_dict(), "./saved_model/saved_model_220628.pt")

        plt.xlabel("epochs")
        plt.ylabel("mean loss")
        plt.plot(range(epochs), mean_loss)
        plt.show()

    if not train:
        model.load_state_dict(torch.load("./saved_model/saved_model_220628.pt"))
    model.eval()

    pred = model(X)
    pred = pred.detach().numpy()
    print("train nmae: ", NMAE(y, pred))

    plt.plot(range(len(X)), y)
    plt.plot(range(len(X)), pred)
    plt.show()

    pred = model(X_test)
    pred = pred.detach().numpy()
    print("test nmae: ", NMAE(y_test, pred))

    plt.plot(range(len(X_test)), y_test)
    plt.plot(range(len(X_test)), pred)
    plt.show()