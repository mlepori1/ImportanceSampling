import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def train_baseline(train_data, epochs, model, optimizer, criterion):
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch)
        for i in range(len(train_data)):
            # get the inputs; data is a list of [inputs, labels]
            data = train_data[i]
            inputs = data['image'].unsqueeze(0)
            classes = data['class'].unsqueeze(0).unsqueeze(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, classes)
            loss.backward()
            optimizer.step()

    print('Finished Training')
    return model