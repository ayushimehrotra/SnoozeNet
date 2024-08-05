import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from torch.cuda.amp import autocast, GradScaler

from dataloader import create_dataloaders, load_data
from utils import clear_mem
from model import make_model


def cross_val(data, num_folds):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    folds = np.array_split(indices, num_folds)
    test_indices = folds[0]
    train_indices = np.setdiff1d(indices, test_indices)

    data_train, data_test = data[train_indices], data[test_indices]
    return data_train, data_test


def find_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Macro F1 Score: {macro_f1}")
    f1_per_class = f1_score(y_true, y_pred, average=None)
    print(f"Per-class F1 scores: {f1_per_class}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    sensitivity = np.diag(cm) / np.sum(cm, axis=1)
    print(f"Sensitivity (mean): {np.mean(sensitivity)}")

    specificity = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp))
    print(f"Specificity (mean): {np.mean(specificity)}")

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa}")


def train_model(data):
    raw_data = []
    num_folds = 10
    num_epochs = 100
    if data == "SleepEDF20":
        raw_data = load_data("datasets/sleep-cassette-EDF")
    elif data == "SleepEDF78":
        raw_data = load_data("datasets/sleep-telemetry-EDF")

    print("Raw Data Loaded")
    for i in range(num_folds):
        print("Fold #" + str(i+1))
        raw_data_train, raw_data_test = cross_val(raw_data, num_folds)
        train_dataloader, test_dataloader = create_dataloaders(raw_data_train, raw_data_test)
        del raw_data_train
        del raw_data_test
        clear_mem()

        transformer = make_model()
        print("Transformer Created")
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

        transformer.train()
        scaler = GradScaler()
        for epoch in range(num_epochs):
            for step, (x, y) in enumerate(train_dataloader):
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                with autocast():
                    output = transformer(x)
                    print(output, y)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

        transformer.eval()
        for step, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                val_output = transformer(x)
                find_metrics(y, val_output)
                loss = criterion(val_output, y)
                print(f"Validation Loss: {loss.item()}")