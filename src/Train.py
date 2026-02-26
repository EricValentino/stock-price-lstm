import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from model import PredictionModel


def load_data(ticker="MSFT"):
    df = yf.download(ticker, "2020-01-01")

    scaler = StandardScaler()
    df["Close"] = scaler.fit_transform(df[["Close"]])

    return df, scaler


def create_sequences(df, seq_length=30):

    data = []

    for i in range(len(df) - seq_length):
        data.append(df.Close[i:i+seq_length])

    data = np.array(data)

    train_size = int(0.8 * len(data))

    X_train = data[:train_size, :-1]
    y_train = data[:train_size, -1]

    return X_train, y_train


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, scaler = load_data()

    X_train, y_train = create_sequences(df)

    X_train = torch.tensor(X_train).float().unsqueeze(-1).to(device)
    y_train = torch.tensor(y_train).float().unsqueeze(-1).to(device)

    model = PredictionModel(1, 32, 2, 1).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 200

    for epoch in range(epochs):

        preds = model(X_train)
        loss = criterion(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(epoch, loss.item())

    torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train()